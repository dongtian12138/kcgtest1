#!/usr/bin/env python3

"""Nominal physical-validity ladder for the D38999 insertion proxy V2.

This runner intentionally does *not* run Bcapture until one nominal episode
has proved the real iiwa -> hand2arm -> handbase -> grasp latch -> Plug load
path.  Robot motion is applied only through iiwa joint commands generated
from TCP targets.  PlugBody and CouplingNut never receive a motion actuator,
pose write, or control force.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import traceback

import numpy as np
import yaml

from kcg_connector.compliant_insertion import (
    ControllerState,
    effective_lateral_posthoc,
    InsertionObservation,
    InsertionState,
    full_seated_posthoc,
    load_compliant_insertion_config,
    step_compliant_insertion,
)
from kcg_connector.d38999_insert_proxy_v2 import load_insert_proxy_v2
from kcg_connector.d38999_physical_insertion import solve_fixed_q7_tcp_pose
from kcg_connector.postgrasp_error_injection import (
    assembly_tcp_from_grasp_tcp,
    integrate_assembly_twist_on_grasp_tcp,
    PostGraspError,
    injection_error,
    measure_error_from_nominal,
)
from kcg_connector.d38999_tabletop_pick import (
    iiwa14_grasp_tcp_transform,
    load_d38999_tabletop_pick_config,
    verify_d38999_pick_dependencies,
)
from kcg_connector.virtual_wrist_ft_runtime import (
    VirtualWristFtMonitor,
    load_virtual_wrist_ft_monitor_config,
    reaction_row_index,
    verify_virtual_wrist_ft_monitor_inputs,
)

from d38999_iiwa_hand_v2_scene import (
    ARM_NAMES,
    ARTICULATION_ROOT,
    COLLISION_PROFILES,
    COUPLING_NUT,
    HAND_BASE,
    PLUG_BODY,
    RECEPTACLE,
    apply_diagnostic_collision_profile,
    author_scene,
    topology_report,
)


ACTIVE_HAND_NAMES = ("f1j1", "f1j2", "f2j1", "f3j2")


class _InhandValidationComplete(Exception):
    """Internal, successful early exit before any contact experiment."""


class _ResponseIdentificationComplete(Exception):
    """Internal, successful exit after bounded contact probes."""


def _arguments(repository: Path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--nominal-only", action="store_true")
    parser.add_argument(
        "--inhand-validation-only",
        action="store_true",
        help="stop after reset-only latch injection and outer-loop preservation checks",
    )
    parser.add_argument(
        "--identify-contact-response",
        action="store_true",
        help="stop after measured-wrench +/-X,+/-Y,+/-Rx,+/-Ry contact probes",
    )
    parser.add_argument(
        "--ladder-case",
        choices=("nominal", "x", "y", "rx", "ry"),
        default="nominal",
    )
    parser.add_argument(
        "--collision-profile",
        choices=COLLISION_PROFILES,
        default="full",
        help="POSTHOC_DIAGNOSTIC collider isolation authored before reset",
    )
    parser.add_argument(
        "--control-mode",
        choices=("rigid", "compliant"),
        default="compliant",
    )
    parser.add_argument(
        "--target-stage",
        choices=("guide_0p25", "additional_1", "depth_3", "depth_6", "full_9"),
        default="guide_0p25",
    )
    parser.add_argument(
        "--precontact-tracking",
        choices=("open_loop", "joint_state_outer_loop"),
        default="joint_state_outer_loop",
    )
    for axis in ("x", "y", "z"):
        parser.add_argument(
            f"--inhand-d{axis}-mm",
            type=float,
            default=None,
            help=f"reset-only Plug-in-hand translation error along {axis.upper()}",
        )
    for axis in ("x", "y", "z"):
        parser.add_argument(
            f"--inhand-dr{axis}-deg",
            type=float,
            default=None,
            help=f"reset-only Plug-in-hand rotation error about {axis.upper()}",
        )
    parser.add_argument(
        "--config",
        default=str(
            repository
            / "src/kcg_connector/config/d38999_capture_sweep_v1.yaml"
        ),
    )
    parser.add_argument(
        "--asset",
        default=str(
            repository
            / "artifacts/kcg_connector/isaac/d38999_insert_proxy_v2.usda"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            repository
            / "artifacts/kcg_connector/d38999_insert_proxy_v2/nominal_physics_v1"
        ),
    )
    parser.add_argument(
        "--screw-after-seat",
        action="store_true",
        help=(
            "after a fully seated 9 mm insertion, rotate the TCP one bounded "
            "150-degree segment about the assembly axis via joint 7 while the "
            "three-finger grip only prevents slip; wrench soft gates apply"
        ),
    )
    parser.add_argument(
        "--visual-chain-report",
        default=None,
        help=(
            "path to a kcg_d38999_visual_chain_v1 report; when present, the "
            "requested in-hand error is taken from stage_t_hp."
            "t_hp_error_vs_nominal_xyz_rpy instead of CLI injection args. "
            "POSTHOC_DIAGNOSTIC_ONLY: the visual estimate never authorizes "
            "control and never feeds the formal insertion contract."
        ),
    )
    result = parser.parse_args()
    if not result.run:
        parser.error("nominal physics ladder requires --run")
    if not result.nominal_only:
        parser.error(
            "Bcapture is locked until nominal physics passes; use --nominal-only"
        )
    return result


def _load_inputs(repository: Path, config_path: Path):
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        document.get("schema_version") != "kcg_d38999_capture_sweep_v1"
        or document.get("enabled") is not True
    ):
        raise ValueError("invalid capture-sweep contract")
    forbidden = set(document["boundaries"]["forbidden_controller_inputs"])
    required = {
        "initial_error_truth",
        "receptacle_pose_truth",
        "physx_contact_normal",
        "collider_identity",
        "contact_point_truth",
        "penetration_depth_truth",
    }
    if forbidden != required:
        raise ValueError("forbidden controller-input boundary changed")
    if document["boundaries"]["standalone_body_or_nut_actuator_allowed"]:
        raise ValueError("standalone actuator must remain forbidden")
    if document["boundaries"]["direct_plug_or_nut_motion_force_allowed"]:
        raise ValueError("direct Plug/Nut motion force must remain forbidden")
    proxy = load_insert_proxy_v2(repository / document["proxy_config"])
    controller = load_compliant_insertion_config(
        repository / document["compliant_controller_config"]
    )
    pick_path = repository / document["robot_pick_config"]
    pick = load_d38999_tabletop_pick_config(pick_path)
    dependencies = verify_d38999_pick_dependencies(
        pick, pick_path, repository
    )
    wrist_config = load_virtual_wrist_ft_monitor_config(
        repository / document["wrist_ft_monitor_config"]
    )
    wrist_inputs = verify_virtual_wrist_ft_monitor_inputs(
        wrist_config, repository
    )
    if wrist_inputs["robot_asset"] != dependencies["robot_asset"]:
        raise RuntimeError("pick and wrist contracts use different robots")
    calibration_path = repository / document["direction_calibration_report"]
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("passed") is not True:
        raise RuntimeError("wrist direction calibration has not passed")
    if not calibration.get("cases") or not all(
        item.get("sign_ok") is True for item in calibration["cases"]
    ):
        raise RuntimeError("direction calibration case evidence is incomplete")
    return (
        document,
        proxy,
        controller,
        pick,
        dependencies,
        wrist_config,
        calibration_path,
        calibration,
    )


def _rotation_angle(rotation: np.ndarray) -> float:
    cosine = max(-1.0, min(1.0, (float(np.trace(rotation)) - 1.0) / 2.0))
    return math.acos(cosine)


def _quat_wxyz_rotation(quaternion) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = (w / norm, x / norm, y / norm, z / norm)
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _pose_from_view(view):
    positions, quaternions = view.get_world_poses()
    return (
        np.asarray(positions[0], dtype=np.float64),
        _quat_wxyz_rotation(quaternions[0]),
    )


def _relative_pose(parent_pose, child_pose):
    parent_position, parent_rotation = parent_pose
    child_position, child_rotation = child_pose
    return (
        parent_rotation.T @ (child_position - parent_position),
        parent_rotation.T @ child_rotation,
    )


def _exp_rotation(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle <= 1.0e-15:
        return np.eye(3)
    x, y, z = vector / angle
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _xyz_from_rotation(rotation: np.ndarray) -> np.ndarray:
    """Return intrinsic XYZ angles for the small frame-audit residual."""

    matrix = np.asarray(rotation, dtype=np.float64)
    ry = math.asin(max(-1.0, min(1.0, -float(matrix[2, 0]))))
    if abs(math.cos(ry)) > 1.0e-9:
        rx = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
        rz = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    else:
        rx = math.atan2(-float(matrix[1, 2]), float(matrix[1, 1]))
        rz = 0.0
    return np.asarray((rx, ry, rz), dtype=np.float64)


def _requested_inhand_error(arguments, document) -> PostGraspError:
    configured = document["post_grasp_error"]
    translation = configured["translation_m"]
    rotation = configured["rotation_rad"]
    translation_m = tuple(
        float(
            getattr(arguments, f"inhand_d{axis}_mm")
            if getattr(arguments, f"inhand_d{axis}_mm") is not None
            else 1000.0 * float(translation[axis])
        )
        / 1000.0
        for axis in ("x", "y", "z")
    )
    rotation_xyz_rad = tuple(
        math.radians(
            float(
                getattr(arguments, f"inhand_dr{axis}_deg")
                if getattr(arguments, f"inhand_dr{axis}_deg") is not None
                else math.degrees(float(rotation[axis]))
            )
        )
        for axis in ("x", "y", "z")
    )
    return PostGraspError(
        translation_m=translation_m,
        rotation_xyz_rad=rotation_xyz_rad,
        source=configured["source"],
    )


def main():
    repository = Path(__file__).resolve().parents[3]
    arguments = _arguments(repository)
    config_path = Path(arguments.config).expanduser().resolve()
    output = Path(arguments.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    (
        document,
        proxy,
        controller_config,
        pick,
        dependencies,
        wrist_config,
        calibration_path,
        calibration,
    ) = _load_inputs(repository, config_path)
    case_spec = document["ladder_cases"][arguments.ladder_case]
    requested_inhand = _requested_inhand_error(arguments, document)
    visual_chain_error = None
    if arguments.visual_chain_report is not None:
        chain_path = Path(arguments.visual_chain_report).expanduser().resolve()
        chain_report = json.loads(chain_path.read_text(encoding="utf-8"))
        stage = chain_report.get("stage_t_hp")
        if stage is None or "t_hp_error_vs_nominal_xyz_rpy" not in stage:
            raise ValueError(
                "visual chain report has no stage_t_hp."
                "t_hp_error_vs_nominal_xyz_rpy"
            )
        delta = np.asarray(
            stage["t_hp_error_vs_nominal_xyz_rpy"], dtype=np.float64
        )
        # The chain's plug frame is Rx(pi) from the sweep's hand-aligned
        # nominal plug frame; convert the right-relative delta accordingly:
        # T_delta_sweep = Rx(pi) @ T_delta_chain @ Rx(pi).
        c2_yaw_estimate_rad = float(delta[5])
        # C2 proxy: the mating face carries no yaw key, so the estimated
        # Plug-frame Rz is unresolved noise and must NOT be injected as a
        # correction.  Zero it; the compliant controller's own bounded yaw
        # search remains the only yaw mechanism.
        # Axial (z) is force-guided by the compliant descent: the visual
        # axial estimate carries a systematic ~3 mm bias, so inject only the
        # force-blind dimensions (x, y, rx, ry).  Same policy family as the
        # C2 yaw zeroing.
        requested_inhand = PostGraspError(
            translation_m=(
                float(delta[0]),
                float(-delta[1]),
                0.0,
            ),
            rotation_xyz_rad=(
                float(delta[3]),
                float(-delta[4]),
                0.0,
            ),
            source="reset_only_simulation",
        )
        visual_chain_error = {
            "report_path": str(chain_path),
            "t_hp_pose_valid": stage.get("pose_valid"),
            "t_hp_pose_valid_reasons": stage.get("pose_valid_reasons"),
            "c2_resolution": stage.get("c2_resolution"),
            "delta_xyz_rpy": delta.tolist(),
            "c2_yaw_estimate_rad": c2_yaw_estimate_rad,
            "axial_z_estimate_m": float(delta[2]),
            "axial_z_injected": False,
            "axial_z_policy": "ZERO_INJECT_FORCE_GUIDED",
            "c2_yaw_injected": False,
            "c2_yaw_policy": "ZERO_INJECT_UNRESOLVED",
            "diagnostic_only": True,
        }
    inhand_nonzero = bool(
        np.linalg.norm(requested_inhand.translation_m) > 0.0
        or np.linalg.norm(requested_inhand.rotation_xyz_rad) > 0.0
    )
    if inhand_nonzero and arguments.ladder_case != "nominal":
        raise ValueError(
            "in-hand error experiments cannot also move the Receptacle ladder case"
        )
    if arguments.ladder_case != "nominal":
        nominal_pass_path = repository / document["nominal_pass_report"]
        nominal_pass = json.loads(
            nominal_pass_path.read_text(encoding="utf-8")
        )
        if nominal_pass.get("nominal_physics_valid") is not True:
            raise RuntimeError(
                "offset ladder cases are locked until nominal passes"
            )

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": not arguments.gui,
            "multi_gpu": False,
            "active_gpu": 0,
            "physics_gpu": 0,
        }
    )
    passed = False
    report = {
        "schema_version": "kcg_d38999_nominal_physics_ladder_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "REAL_IIWA_HAND_LOAD_PATH",
        "ladder_case": arguments.ladder_case,
        "collision_profile": arguments.collision_profile,
        "collision_profile_scope": "POSTHOC_DIAGNOSTIC",
        "control_mode": arguments.control_mode,
        "target_stage": arguments.target_stage,
        "precontact_tracking": arguments.precontact_tracking,
        "authored_error_posthoc_only": case_spec,
        "visual_chain_estimate": visual_chain_error,
        "grasp_mode": "GRASP_LATCH_PROXY",
        "requested_inhand_error": {
            "translation_m": list(requested_inhand.translation_m),
            "rotation_xyz_rad": list(requested_inhand.rotation_xyz_rad),
            "source": requested_inhand.source,
        },
        "calibration_report": str(calibration_path.relative_to(repository)),
        "calibration_passed": True,
        "calibration_summary": {
            "case_count": len(calibration["cases"]),
            "all_signs_ok": all(item["sign_ok"] for item in calibration["cases"]),
            "maximum_relative_error": max(item["relative_error"] for item in calibration["cases"]),
            "maximum_cross_axis_error": max(item["cross_axis_error"] for item in calibration["cases"]),
            "maximum_moment_arm_error_nm": max(item["moment_arm_error"] for item in calibration["cases"]),
        },
        "controller_inputs": document["boundaries"]["controller_inputs"],
        "forbidden_controller_inputs": document["boundaries"]["forbidden_controller_inputs"],
        "object_truth_use": (
            "evidence_side_channel_and_posthoc_physics_integrity_scoring_only"
        ),
        "object_truth_can_modify_motion_command": False,
        "object_truth_can_select_terminal_state": False,
        "standalone_actuator_present": False,
        "direct_plug_or_nut_motion_force": False,
        "plug_or_nut_pose_write_after_physics_start": False,
        "contact_normal_used": False,
        "collider_identity_used": False,
        "stages": [],
        "passed": False,
    }
    trace_stream = (output / "steps.jsonl").open("w", encoding="utf-8")
    try:
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import RigidPrim, SingleArticulation
        from isaacsim.core.utils.stage import (
            add_reference_to_stage,
            get_current_stage,
        )
        from isaacsim.core.utils.types import ArticulationAction
        from omni.physx import get_physx_simulation_interface
        from pxr import (
            Gf,
            PhysxSchema,
            PhysicsSchemaTools,
            Sdf,
            UsdGeom,
            UsdLux,
            UsdPhysics,
        )

        rate = int(document["experiment"]["physics_rate_hz"])
        dt = 1.0 / rate
        gates = document["sim_debug_gates"]
        nominal = document["nominal_scene"]
        base_arm = np.asarray(nominal["initial_arm_rad"], dtype=np.float64)
        latch_offset = float(nominal["latch_tcp_to_plug_body_m"])
        grasp_to_assembly = np.asarray(
            (0.0, 0.0, latch_offset), dtype=np.float64
        )
        base_tcp = np.asarray(
            iiwa14_grasp_tcp_transform(tuple(float(value) for value in base_arm)),
            dtype=np.float64,
        )
        initial_tcp = np.asarray(
            iiwa14_grasp_tcp_transform((0.0,) * 7), dtype=np.float64
        )
        global_step = 0

        def build_scene(include_payload: bool):
            World.clear_instance()
            omni.usd.get_context().new_stage()
            app.update()
            world = World(
                stage_units_in_meters=1.0,
                physics_dt=dt,
                rendering_dt=1.0 / 60.0,
                backend="numpy",
                device="cpu",
            )
            stage = get_current_stage()
            author = author_scene(
                stage=stage,
                robot_asset=dependencies["robot_asset"],
                v2_asset=Path(arguments.asset),
                arm_rad=base_arm,
                tcp_transform=base_tcp,
                initial_tcp_transform=initial_tcp,
                proxy=proxy,
                add_reference_to_stage=add_reference_to_stage,
                Gf=Gf,
                Sdf=Sdf,
                UsdGeom=UsdGeom,
                UsdPhysics=UsdPhysics,
                include_payload=include_payload,
                latch_tcp_to_body_m=latch_offset,
                post_grasp_error=requested_inhand if include_payload else None,
                preinsert_gap_m=proxy.preinsert_gap if include_payload else None,
                receptacle_error_translation_assembly_m=case_spec[
                    "translation_assembly_m"
                ],
                receptacle_error_rotation_xyz_rad=case_spec[
                    "rotation_xyz_rad"
                ],
            )
            collision_profile = None
            if include_payload:
                collision_profile = apply_diagnostic_collision_profile(
                    stage, UsdPhysics, arguments.collision_profile
                )
                for body_path in (PLUG_BODY, COUPLING_NUT, RECEPTACLE):
                    report_api = PhysxSchema.PhysxContactReportAPI.Apply(
                        stage.GetPrimAtPath(body_path)
                    )
                    report_api.CreateThresholdAttr().Set(0.0)
                if arguments.gui:
                    from isaacsim.core.rendering_manager import ViewportManager

                    lights = UsdGeom.Xform.Define(
                        stage, "/World/V2NominalGuiLighting"
                    )
                    dome = UsdLux.DomeLight.Define(
                        stage, str(lights.GetPath()) + "/Fill"
                    )
                    dome.CreateIntensityAttr(700.0)
                    key = UsdLux.DistantLight.Define(
                        stage, str(lights.GetPath()) + "/Key"
                    )
                    key.CreateIntensityAttr(1800.0)
                    UsdGeom.Xformable(key).AddRotateXYZOp().Set(
                        Gf.Vec3f(-35.0, 25.0, 20.0)
                    )
                    target = np.asarray(author["root_position_world_m"])
                    ViewportManager.set_camera_view(
                        camera="/OmniverseKit_Persp",
                        eye=target + np.asarray((0.16, 0.20, 0.15)),
                        target=target + np.asarray((0.0, 0.0, 0.015)),
                    )
                    app.update()
            robot = world.scene.add(
                SingleArticulation(
                    prim_path=ARTICULATION_ROOT,
                    name="d38999_v2_nominal_handarm",
                )
            )
            world.reset()
            world.get_physics_context().set_gravity(-9.81)
            if not robot.handles_initialized:
                raise RuntimeError("iiwa-hand articulation failed to initialize")
            dof_map = {name: index for index, name in enumerate(robot.dof_names)}
            required = set(ARM_NAMES + ACTIVE_HAND_NAMES)
            if not required.issubset(dof_map):
                raise RuntimeError(f"missing robot DOFs: {sorted(required-set(dof_map))}")
            arm_indices = np.asarray([dof_map[name] for name in ARM_NAMES], dtype=np.int32)
            hand_indices = np.asarray([dof_map[name] for name in ACTIVE_HAND_NAMES], dtype=np.int32)
            controlled = np.concatenate((arm_indices, hand_indices))
            target = np.concatenate((base_arm, np.asarray((1.0, 0.0, 0.0, 0.0)))).astype(np.float32)
            controller = robot.get_articulation_controller()
            kps = np.zeros(robot.num_dof, dtype=np.float32)
            kds = np.zeros(robot.num_dof, dtype=np.float32)
            kps[arm_indices] = pick.robot.arm_stiffness
            kds[arm_indices] = pick.robot.arm_damping
            kps[hand_indices] = pick.robot.hand_stiffness
            kds[hand_indices] = pick.robot.hand_damping
            controller.set_gains(kps=kps, kds=kds, save_to_usd=False)
            metadata = dict(robot._articulation_view._metadata.joint_indices)
            row = reaction_row_index(metadata, wrist_config)
            result = {
                "world": world,
                "stage": stage,
                "robot": robot,
                "row": row,
                "arm_indices": arm_indices,
                "hand_indices": hand_indices,
                "controlled": controlled,
                "target": target,
                "authoring": author,
                "topology": None,
                "body_view": None,
                "nut_view": None,
                "receptacle_view": None,
                "collision_profile": collision_profile,
                "arm_command_bias": np.zeros(7, dtype=np.float64),
            }
            if include_payload:
                body = RigidPrim(
                    prim_paths_expr=PLUG_BODY,
                    name="d38999_v2_nominal_plug_body",
                    reset_xform_properties=False,
                )
                nut = RigidPrim(
                    prim_paths_expr=COUPLING_NUT,
                    name="d38999_v2_nominal_coupling_nut",
                    reset_xform_properties=False,
                )
                receptacle = RigidPrim(
                    prim_paths_expr=RECEPTACLE,
                    name="d38999_v2_nominal_receptacle",
                    reset_xform_properties=False,
                )
                body.initialize()
                nut.initialize()
                receptacle.initialize()
                if (
                    not body.is_physics_handle_valid()
                    or not nut.is_physics_handle_valid()
                    or not receptacle.is_physics_handle_valid()
                ):
                    raise RuntimeError("V2 payload physics views are invalid")
                result["body_view"] = body
                result["nut_view"] = nut
                result["receptacle_view"] = receptacle
                result["topology"] = topology_report(stage, UsdPhysics)
                result["topology"]["collision_profile"] = collision_profile
                if result["topology"]["world_to_plug_fixed_joints"]:
                    raise RuntimeError("world-to-Plug FixedJoint is forbidden")
            return result

        def raw_sample(scene):
            values = np.asarray(
                scene["robot"].get_measured_joint_forces(
                    joint_indices=np.asarray([scene["row"]], dtype=np.int32)
                ),
                dtype=np.float64,
            )
            if values.shape != (1, 6) or not np.all(np.isfinite(values)):
                raise RuntimeError("invalid hand2arm reaction wrench")
            return values[0]

        def command(scene, positions):
            positions = np.asarray(positions, dtype=np.float64).copy()
            if arguments.precontact_tracking == "joint_state_outer_loop":
                positions[:7] += scene["arm_command_bias"]
            scene["robot"].apply_action(
                ArticulationAction(
                    joint_positions=np.asarray(positions, dtype=np.float32),
                    joint_indices=scene["controlled"],
                )
            )

        def move_to_base(scene):
            start = np.asarray(scene["robot"].get_joint_positions(), dtype=np.float64)[scene["controlled"]]
            for index in range(1, 961):
                fraction = index / 960.0
                blend = fraction**3 * (10.0 + fraction * (-15.0 + 6.0 * fraction))
                command(scene, start + blend * (scene["target"] - start))
                scene["world"].step(render=arguments.gui and index % 4 == 0)
            for _ in range(120):
                command(scene, scene["target"])
                scene["world"].step(render=arguments.gui)
            if arguments.precontact_tracking == "joint_state_outer_loop":
                # The USD drives need a small gravity/load feed-forward offset.
                # Integrate only robot joint-state error in free space; neither
                # Plug nor Receptacle pose enters this correction.
                stable = 0
                for _ in range(960):
                    measured = np.asarray(
                        scene["robot"].get_joint_positions(), dtype=np.float64
                    )[scene["arm_indices"]]
                    error = base_arm - measured
                    scene["arm_command_bias"] = np.clip(
                        scene["arm_command_bias"] + 0.035 * error,
                        -0.02,
                        0.02,
                    )
                    command(scene, scene["target"])
                    scene["world"].step(render=arguments.gui)
                    stable = stable + 1 if np.max(np.abs(error)) <= 2.0e-5 else 0
                    if stable >= 60:
                        break
            measured = np.asarray(scene["robot"].get_joint_positions(), dtype=np.float64)[scene["arm_indices"]]
            return float(np.max(np.abs(measured - base_arm)))

        task_origin_base = base_tcp[:3, 3] + base_tcp[:3, :3] @ np.asarray((0.0, 0.0, latch_offset))
        empty = build_scene(False)
        empty_error = move_to_base(empty)
        monitor = VirtualWristFtMonitor(
            wrist_config,
            reaction_row=empty["row"],
            task_origin_world=task_origin_base,
            task_z_axis_world=base_tcp[:3, 2],
        )
        empty_raw = []
        for _ in range(wrist_config.home_tare_window_steps):
            command(empty, empty["target"])
            empty["world"].step(render=arguments.gui)
            global_step += 1
            q = np.asarray(empty["robot"].get_joint_positions(), dtype=np.float64)[empty["arm_indices"]]
            tcp = np.asarray(iiwa14_grasp_tcp_transform(tuple(float(v) for v in q)))
            sensor_position = tcp[:3, 3] - tcp[:3, :3] @ np.asarray((0.0, 0.0, 0.4))
            raw = raw_sample(empty)
            empty_raw.append(raw)
            monitor.observe(
                raw,
                global_step=global_step,
                runtime_phase="initial_settle",
                sensor_position_world=sensor_position,
                sensor_rotation_world=tcp[:3, :3],
            )
        home_baseline = monitor.capture_home_tare()

        loaded = build_scene(True)
        if loaded["row"] != empty["row"]:
            raise RuntimeError("hand2arm reaction row changed with payload")
        loaded_error = move_to_base(loaded)
        report["topology"] = loaded["topology"]
        report["authoring"] = loaded["authoring"]
        report["hand2arm_reaction_row"] = loaded["row"]
        report["post_move_arm_tracking_error_rad"] = {
            "empty": empty_error,
            "loaded": loaded_error,
        }
        report["joint_state_outer_loop"] = {
            "enabled": arguments.precontact_tracking == "joint_state_outer_loop",
            "controller_inputs": ["robot_joint_state", "robot_tcp_fk"],
            "object_truth_used": False,
            "empty_command_bias_rad": empty["arm_command_bias"].tolist(),
            "loaded_command_bias_rad": loaded["arm_command_bias"].tolist(),
        }

        # POSTHOC_SIM_TRUTH: verify only that the reset-authored fixed latch
        # contains the requested error.  This value is never passed to command,
        # IK, compliant control, or a state transition.
        measured_q = np.asarray(
            loaded["robot"].get_joint_positions(), dtype=np.float64
        )[loaded["arm_indices"]]
        measured_tcp = np.asarray(
            iiwa14_grasp_tcp_transform(
                tuple(float(value) for value in measured_q)
            )
        )
        measured_hand_pose = (
            measured_tcp[:3, 3]
            - measured_tcp[:3, :3] @ np.asarray((0.0, 0.0, 0.4)),
            measured_tcp[:3, :3],
        )
        measured_hand_body = _relative_pose(
            measured_hand_pose, _pose_from_view(loaded["body_view"])
        )
        actual_inhand = measure_error_from_nominal(
            measured_hand_body[0],
            measured_hand_body[1],
            (0.0, 0.0, float(0.4 + latch_offset)),
        )
        inhand_difference = injection_error(requested_inhand, actual_inhand)
        injection_config = document["post_grasp_error"]
        injection_valid = bool(
            inhand_difference["translation_error_norm_m"]
            <= float(injection_config["maximum_injection_translation_error_m"])
            and inhand_difference["rotation_error_norm_rad"]
            <= float(injection_config["maximum_injection_rotation_error_rad"])
        )
        report["post_grasp_error_injection"] = {
            "stage": "POST_GRASP_ERROR_INJECTION_VALID",
            "frame": injection_config["frame"],
            "requested": report["requested_inhand_error"],
            "posthoc_actual": {
                "translation_m": list(actual_inhand.translation_m),
                "rotation_xyz_rad": list(actual_inhand.rotation_xyz_rad),
            },
            "requested_vs_actual_difference": inhand_difference,
            "translation_error_gate_m": float(
                injection_config["maximum_injection_translation_error_m"]
            ),
            "rotation_error_gate_rad": float(
                injection_config["maximum_injection_rotation_error_rad"]
            ),
            "joint_state_outer_loop_completed_before_measurement": (
                arguments.precontact_tracking == "joint_state_outer_loop"
            ),
            "object_truth_controller_input": False,
            "passed": injection_valid,
        }
        report["joint_state_outer_loop"]["preserved_inhand_error"] = (
            injection_valid
        )
        if not injection_valid:
            raise RuntimeError("INVALID_INHAND_ERROR_INJECTION")
        if arguments.inhand_validation_only:
            report["mode"] = "POST_GRASP_ERROR_INJECTION_VALIDATION"
            report["nominal_physics_valid"] = None
            report["full_nominal_seated"] = False
            report["bcapture_scan_run"] = False
            report["bcapture_lock_reason"] = "INJECTION_VALIDATION_ONLY"
            report["passed"] = True
            passed = True
            raise _InhandValidationComplete()

        reference_hand_body = None
        reference_body_nut = None
        previous_tcp_position = None
        diagnostics = {
            "maximum_tcp_step_m": 0.0,
            "total_absolute_tcp_path_m": 0.0,
            "maximum_tcp_linear_velocity_m_s": 0.0,
            "maximum_plug_linear_velocity_m_s": 0.0,
            "maximum_plug_angular_velocity_rad_s": 0.0,
            "maximum_nut_linear_velocity_m_s": 0.0,
            "maximum_nut_angular_velocity_rad_s": 0.0,
            "maximum_payload_kinetic_energy_j": 0.0,
            "maximum_plug_hand_translation_drift_m": 0.0,
            "maximum_plug_hand_rotation_drift_rad": 0.0,
            "maximum_nut_plug_translation_drift_m": 0.0,
            "physics_invalid": False,
            "first_invalid_reason": None,
            "maximum_wrench_sample_age_s": 0.0,
            "stale_wrench_detected": False,
        }
        posthoc_first_contacts = {}
        loaded_raw = []

        def posthoc_v2_contact_pairs():
            """Return current V2 pair identities for diagnostics only."""

            headers, _, _ = (
                get_physx_simulation_interface().get_full_contact_report()
            )
            found = []
            plug_roots = (PLUG_BODY, COUPLING_NUT)
            for header in headers:
                paths = (
                    str(PhysicsSchemaTools.intToSdfPath(header.actor0)),
                    str(PhysicsSchemaTools.intToSdfPath(header.actor1)),
                    str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                    str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
                )
                has_plug = any(
                    any(path.startswith(root) for root in plug_roots)
                    for path in paths
                )
                has_receptacle = any(
                    path.startswith(RECEPTACLE) for path in paths
                )
                if has_plug and has_receptacle:
                    found.append(
                        {
                            "paths": list(paths),
                            "contact_record_count": int(
                                header.num_contact_data
                            ),
                        }
                    )
            return found

        def observe_loaded(runtime_phase: str, command_arm, *, trace=True):
            nonlocal global_step, previous_tcp_position, reference_hand_body, reference_body_nut
            target = np.concatenate((np.asarray(command_arm, dtype=np.float64), loaded["target"][7:]))
            command(loaded, target)
            loaded["world"].step(render=arguments.gui and global_step % 4 == 0)
            global_step += 1
            q = np.asarray(loaded["robot"].get_joint_positions(), dtype=np.float64)[loaded["arm_indices"]]
            tcp = np.asarray(iiwa14_grasp_tcp_transform(tuple(float(value) for value in q)))
            task_origin = tcp[:3, 3] + tcp[:3, :3] @ np.asarray((0.0, 0.0, latch_offset))
            sensor_position = tcp[:3, 3] - tcp[:3, :3] @ np.asarray((0.0, 0.0, 0.4))
            monitor.task_origin_world = task_origin
            monitor.task_rotation_world = tcp[:3, :3]
            raw = raw_sample(loaded)
            observed = monitor.observe(
                raw,
                global_step=global_step,
                runtime_phase=runtime_phase,
                sensor_position_world=sensor_position,
                sensor_rotation_world=tcp[:3, :3],
            )
            body_pose = _pose_from_view(loaded["body_view"])
            nut_pose = _pose_from_view(loaded["nut_view"])
            receptacle_pose = _pose_from_view(loaded["receptacle_view"])
            hand_pose = (
                tcp[:3, 3] - tcp[:3, :3] @ np.asarray((0.0, 0.0, 0.4)),
                tcp[:3, :3],
            )
            hand_body = _relative_pose(hand_pose, body_pose)
            body_nut = _relative_pose(body_pose, nut_pose)
            if reference_hand_body is None:
                reference_hand_body = (hand_body[0].copy(), hand_body[1].copy())
                reference_body_nut = (body_nut[0].copy(), body_nut[1].copy())
            plug_hand_translation = float(np.linalg.norm(hand_body[0] - reference_hand_body[0]))
            plug_hand_rotation = _rotation_angle(reference_hand_body[1].T @ hand_body[1])
            nut_plug_translation = float(np.linalg.norm(body_nut[0] - reference_body_nut[0]))
            plug_linear = float(np.linalg.norm(loaded["body_view"].get_linear_velocities()[0]))
            plug_angular = float(np.linalg.norm(loaded["body_view"].get_angular_velocities()[0]))
            nut_linear = float(np.linalg.norm(loaded["nut_view"].get_linear_velocities()[0]))
            nut_angular = float(np.linalg.norm(loaded["nut_view"].get_angular_velocities()[0]))
            kinetic = 0.5 * proxy.physics.plug_body_mass_kg * plug_linear**2
            kinetic += 0.5 * proxy.physics.coupling_nut_mass_kg * nut_linear**2
            kinetic += 0.5 * max(proxy.physics.plug_body_diagonal_inertia_kg_m2) * plug_angular**2
            kinetic += 0.5 * max(proxy.physics.coupling_nut_diagonal_inertia_kg_m2) * nut_angular**2
            tcp_step = 0.0 if previous_tcp_position is None else float(np.linalg.norm(tcp[:3, 3] - previous_tcp_position))
            previous_tcp_position = tcp[:3, 3].copy()
            diagnostics["maximum_tcp_step_m"] = max(diagnostics["maximum_tcp_step_m"], tcp_step)
            diagnostics["total_absolute_tcp_path_m"] += tcp_step
            diagnostics["maximum_tcp_linear_velocity_m_s"] = max(diagnostics["maximum_tcp_linear_velocity_m_s"], tcp_step / dt)
            diagnostics["maximum_plug_linear_velocity_m_s"] = max(diagnostics["maximum_plug_linear_velocity_m_s"], plug_linear)
            diagnostics["maximum_plug_angular_velocity_rad_s"] = max(diagnostics["maximum_plug_angular_velocity_rad_s"], plug_angular)
            diagnostics["maximum_nut_linear_velocity_m_s"] = max(diagnostics["maximum_nut_linear_velocity_m_s"], nut_linear)
            diagnostics["maximum_nut_angular_velocity_rad_s"] = max(diagnostics["maximum_nut_angular_velocity_rad_s"], nut_angular)
            diagnostics["maximum_payload_kinetic_energy_j"] = max(diagnostics["maximum_payload_kinetic_energy_j"], kinetic)
            diagnostics["maximum_plug_hand_translation_drift_m"] = max(diagnostics["maximum_plug_hand_translation_drift_m"], plug_hand_translation)
            diagnostics["maximum_plug_hand_rotation_drift_rad"] = max(diagnostics["maximum_plug_hand_rotation_drift_rad"], plug_hand_rotation)
            diagnostics["maximum_nut_plug_translation_drift_m"] = max(diagnostics["maximum_nut_plug_translation_drift_m"], nut_plug_translation)
            compensated = observed.get("compensated_wrench_task")
            record = {
                "step": global_step,
                "phase": runtime_phase,
                "tcp_position_world_m": tcp[:3, 3].tolist(),
                "raw_parent_on_child": raw.tolist(),
                "compensated_environment_on_tool_assembly": compensated,
                "tcp_step_m": tcp_step,
                "plug_hand_translation_drift_m": plug_hand_translation,
                "plug_hand_rotation_drift_rad": plug_hand_rotation,
                "nut_plug_translation_drift_m": nut_plug_translation,
                "plug_linear_velocity_m_s": plug_linear,
                "plug_angular_velocity_rad_s": plug_angular,
                "nut_linear_velocity_m_s": nut_linear,
                "nut_angular_velocity_rad_s": nut_angular,
                "payload_kinetic_energy_j": kinetic,
                "posthoc_plug_body_position_world_m": body_pose[0].tolist(),
                "posthoc_receptacle_position_world_m": receptacle_pose[0].tolist(),
            }
            pairs = posthoc_v2_contact_pairs()
            if pairs and runtime_phase not in posthoc_first_contacts:
                body_in_receptacle, body_rotation_in_receptacle = (
                    _relative_pose(receptacle_pose, body_pose)
                )
                wrench = np.asarray(
                    compensated if compensated is not None else (0.0,) * 6,
                    dtype=np.float64,
                )
                posthoc_first_contacts[runtime_phase] = {
                    "label": "POSTHOC_DIAGNOSTIC",
                    "controller_input": False,
                    "step": global_step,
                    "phase": runtime_phase,
                    "pairs": pairs,
                    "plug_body_origin_in_receptacle_m": (
                        body_in_receptacle.tolist()
                    ),
                    "plug_rotation_in_receptacle": (
                        body_rotation_in_receptacle.tolist()
                    ),
                    "wrench_assembly": wrench.tolist(),
                    "tcp_position_world_m": tcp[:3, 3].tolist(),
                    "tcp_rotation_world": tcp[:3, :3].tolist(),
                    "tcp_step_m": tcp_step,
                    "plug_linear_velocity_m_s": plug_linear,
                    "plug_angular_velocity_rad_s": plug_angular,
                    "payload_kinetic_energy_j": kinetic,
                    "solver_transient": bool(
                        tcp_step > float(gates["minimum_max_step_m"])
                        or plug_linear
                        > float(gates["maximum_linear_velocity_m_s"])
                        or kinetic
                        > float(gates["maximum_kinetic_energy_j"])
                    ),
                }
            if trace:
                trace_stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            return {
                "q": q,
                "tcp": tcp,
                "raw": raw,
                "observed": observed,
                "body_pose": body_pose,
                "nut_pose": nut_pose,
                "receptacle_pose": receptacle_pose,
                "record": record,
            }

        # Capture the actual payload baseline in this run and this scene.
        payload_samples = []
        for _ in range(wrist_config.payload_baseline_window_steps):
            sample = observe_loaded("unsupported_final_hold", base_arm)
            loaded_raw.append(sample["raw"])
            payload_samples.append(sample)
        payload_baseline = monitor.capture_payload_baseline()
        report["wrench_baselines"] = {
            "w_parent_on_child_raw_empty_mean": np.mean(np.asarray(empty_raw), axis=0).tolist(),
            "w_parent_on_child_raw_loaded_mean": np.mean(np.asarray(loaded_raw), axis=0).tolist(),
            "home_empty_baseline_canonical": home_baseline,
            "payload_baseline_canonical": payload_baseline,
            "w_environment_on_tool_relation": "-(raw_parent_on_child - no_contact_baseline_raw)",
        }

        initial_body_pose = payload_samples[-1]["body_pose"]
        initial_tcp_measured = payload_samples[-1]["tcp"]
        initial_axis = base_tcp[:3, 2].copy()
        initial_axis /= np.linalg.norm(initial_axis)
        initial_receptacle_pose = payload_samples[-1]["receptacle_pose"]
        initial_body_in_receptacle = _relative_pose(
            initial_receptacle_pose, initial_body_pose
        )
        initial_rpy_error = _xyz_from_rotation(
            initial_body_in_receptacle[1]
        )
        fk_assembly_position = (
            initial_tcp_measured[:3, 3]
            + initial_tcp_measured[:3, :3]
            @ np.asarray((0.0, 0.0, latch_offset))
        )
        ideal_relative = np.asarray((0.0, 0.0, -proxy.preinsert_gap))
        report["nominal_frame_audit"] = {
            "label": "POSTHOC_SIM_TRUTH_FRAME_AUDIT",
            "controller_input": False,
            "plug_mating_frame": {
                "prim": PLUG_BODY,
                "definition": "PlugBody origin at leading mating face",
            },
            "receptacle_mating_frame": {
                "prim": RECEPTACLE,
                "definition": "Receptacle origin at mouth center",
            },
            "assembly_tcp": {
                "definition": "FK grasp_tcp plus fixed latch offset",
                "position_world_m": fk_assembly_position.tolist(),
                "rotation_world": initial_tcp_measured[:3, :3].tolist(),
            },
            "grasp_latch_proxy": {
                "tcp_to_plug_body_translation_m": [
                    0.0,
                    0.0,
                    latch_offset,
                ],
                "rotation_xyz_rad": [0.0, 0.0, 0.0],
            },
            "ideal_T_receptacle_plug": {
                "translation_m": ideal_relative.tolist(),
                "rotation_xyz_rad": [0.0, 0.0, 0.0],
            },
            "measured_posthoc_T_receptacle_plug": {
                "translation_m": initial_body_in_receptacle[0].tolist(),
                "rotation_xyz_rad": initial_rpy_error.tolist(),
            },
            "initial_error": {
                "ex_m": float(initial_body_in_receptacle[0][0]),
                "ey_m": float(initial_body_in_receptacle[0][1]),
                "ez_from_ideal_m": float(
                    initial_body_in_receptacle[0][2] - ideal_relative[2]
                ),
                "erx_rad": float(initial_rpy_error[0]),
                "ery_rad": float(initial_rpy_error[1]),
                "erz_rad": float(initial_rpy_error[2]),
            },
            "fk_assembly_to_physical_plug_translation_m": (
                initial_receptacle_pose[1].T
                @ (initial_body_pose[0] - fk_assembly_position)
            ).tolist(),
            "geometry_centers_local_m": {
                "plug_nose_visual_and_collision": [
                    0.0,
                    0.0,
                    -0.5 * proxy.plug.nose_chamfer_length,
                ],
                "plug_guide_visual_and_collision": [
                    0.0,
                    0.0,
                    -0.5
                    * (proxy.plug.guide_length + proxy.plug.nose_chamfer_length),
                ],
                "receptacle_entry_visual_and_collision": [
                    0.0,
                    0.0,
                    0.5 * proxy.receptacle.entrance_chamfer_length,
                ],
                "receptacle_guide_visual_and_collision": [
                    0.0,
                    0.0,
                    0.5
                    * (
                        proxy.receptacle.guide_length
                        + proxy.receptacle.entrance_chamfer_length
                    ),
                ],
                "separate_visual_collision_center_offset_m": 0.0,
            },
        }
        static_tcp_positions = []
        static_tcp_rotations = []
        for _ in range(int(round(float(gates["static_window_s"]) * rate))):
            sample = observe_loaded("unsupported_final_hold", base_arm)
            static_tcp_positions.append(sample["tcp"][:3, 3].copy())
            static_tcp_rotations.append(sample["tcp"][:3, :3].copy())
        static_translation_drift = max(
            float(np.linalg.norm(position - static_tcp_positions[0]))
            for position in static_tcp_positions
        )
        static_rotation_drift = max(
            _rotation_angle(static_tcp_rotations[0].T @ rotation)
            for rotation in static_tcp_rotations
        )
        static_pass = bool(
            static_translation_drift <= float(gates["maximum_static_tcp_translation_drift_m"])
            and static_rotation_drift <= float(gates["maximum_static_tcp_rotation_drift_rad"])
            and diagnostics["maximum_plug_hand_translation_drift_m"] <= float(gates["maximum_plug_hand_translation_drift_m"])
            and diagnostics["maximum_plug_hand_rotation_drift_rad"] <= float(gates["maximum_plug_hand_rotation_drift_rad"])
        )
        report["stages"].append({
            "stage": "IIWA_HAND_V2_NO_CONTACT_STATIC",
            "passed": static_pass,
            "tcp_translation_drift_m": static_translation_drift,
            "tcp_rotation_drift_rad": static_rotation_drift,
            "plug_hand_translation_drift_m": diagnostics["maximum_plug_hand_translation_drift_m"],
            "plug_hand_rotation_drift_rad": diagnostics["maximum_plug_hand_rotation_drift_rad"],
        })
        if not static_pass:
            raise RuntimeError("PHYSICS_INVALID_STATIC_DRIFT")

        command_arm = base_arm.copy()
        command_tcp = base_tcp.copy()

        def move_tcp(target_tcp, duration_s, phase, expected_speed_m_s):
            nonlocal command_arm, command_tcp
            start_arm = command_arm.copy()
            solved = np.asarray(
                solve_fixed_q7_tcp_pose(
                    tuple(float(value) for value in start_arm),
                    tuple(float(value) for value in target_tcp[:3, 3]),
                    target_rotation=target_tcp[:3, :3],
                ),
                dtype=np.float64,
            )
            steps = max(1, int(round(duration_s * rate)))
            samples = []
            for index in range(1, steps + 1):
                fraction = index / steps
                blend = fraction**3 * (10.0 + fraction * (-15.0 + 6.0 * fraction))
                arm = start_arm + blend * (solved - start_arm)
                sample = observe_loaded(phase, arm)
                samples.append(sample)
                allowed_step = max(
                    float(gates["expected_motion_multiplier"]) * expected_speed_m_s * dt,
                    float(gates["minimum_max_step_m"]),
                )
                if sample["record"]["tcp_step_m"] > allowed_step:
                    raise RuntimeError("PHYSICS_INVALID_TCP_STEP")
            command_arm = solved
            command_tcp = target_tcp.copy()
            return samples

        # 1 mm symmetric free-space test.  It is deliberately below the
        # 12 mm preinsert gap, so contact is not needed to judge this stage.
        no_contact_start_path = diagnostics["total_absolute_tcp_path_m"]
        forward_tcp = command_tcp.copy()
        forward_tcp[:3, 3] += initial_axis * 0.001
        forward_samples = move_tcp(
            forward_tcp,
            1.5,
            "mixed_grip_preinsert_no_contact_forward",
            0.00125,
        )
        backward_samples = move_tcp(
            base_tcp,
            1.5,
            "mixed_grip_preinsert_no_contact_backward",
            0.00125,
        )
        free_wrenches = [
            np.asarray(item["observed"].get("compensated_wrench_task", (0.0,) * 6))
            for item in forward_samples + backward_samples
        ]
        maximum_free_force = max(float(np.linalg.norm(item[:3])) for item in free_wrenches)
        no_contact_path = diagnostics["total_absolute_tcp_path_m"] - no_contact_start_path
        no_contact_pass = bool(
            maximum_free_force <= float(gates["maximum_no_contact_force_n"])
            and no_contact_path <= 0.003
        )
        report["stages"].append({
            "stage": "NO_CONTACT_FORWARD_BACK",
            "passed": no_contact_pass,
            "commanded_each_way_m": 0.001,
            "total_absolute_path_m": no_contact_path,
            "maximum_compensated_force_n": maximum_free_force,
        })
        if not no_contact_pass:
            raise RuntimeError("NO_CONTACT_MOTION_OR_WRENCH_INVALID")

        external_cases = [
            item for item in calibration["cases"]
            if item["orientation"] == "NOMINAL_ORIENTATION" and item["name"] in {"+X", "+Y", "+Z", "OFFSET_PLUS_Y_AT_PLUS_X"}
        ]
        external_pass = bool(len(external_cases) == 4 and all(item["sign_ok"] for item in external_cases))
        report["stages"].append({
            "stage": "KNOWN_EXTERNAL_LOAD_TO_HAND2ARM",
            "passed": external_pass,
            "source": str(calibration_path.relative_to(repository)),
            "cases": external_cases,
        })
        if not external_pass:
            raise RuntimeError("KNOWN_EXTERNAL_LOAD_CALIBRATION_INVALID")

        # Deliberately offset 3 mm in assembly +X so that the nose meets the
        # solid receptacle rim.  This is the ladder's planar/rim contact test,
        # not the nominal insertion.  Both offset and approach are robot TCP
        # commands; the stop condition below is wrist wrench only.
        plane_lateral_tcp = base_tcp.copy()
        plane_lateral_tcp[:3, 3] += base_tcp[:3, 0] * 0.003
        move_tcp(
            plane_lateral_tcp,
            3.0,
            "mixed_grip_preinsert_plane_contact_lateral",
            0.001875,
        )
        plane_coarse_tcp = plane_lateral_tcp.copy()
        plane_coarse_tcp[:3, 3] += initial_axis * 0.008
        plane_coarse = move_tcp(
            plane_coarse_tcp,
            8.0,
            "mixed_grip_preinsert_plane_contact_coarse",
            0.001875,
        )
        maximum_plane_coarse_force = max(
            float(np.linalg.norm(np.asarray(item["observed"]["compensated_wrench_task"])[:3]))
            for item in plane_coarse
        )
        if maximum_plane_coarse_force > float(controller_config["contact_classifier"]["axial_contact_n"]):
            raise RuntimeError("EARLY_CONTACT_DURING_PLANE_COARSE_APPROACH")

        # Wrench-guarded receptacle-rim approach.  Stop only from the measured
        # hand2arm signal; no contact normal, collider identity, or object
        # state enters the stop decision.
        contact_threshold = float(controller_config["contact_classifier"]["axial_contact_n"])
        hard = controller_config["safety"]
        guard_speed = float(controller_config["motion"]["axial_speed_m_s"])
        face_contact = None
        contact_command_position = command_tcp[:3, 3].copy()
        maximum_guard_steps = int(round(25.0 * rate))
        for _ in range(maximum_guard_steps):
            contact_command_position += initial_axis * guard_speed * dt
            command_tcp[:3, 3] = contact_command_position
            command_arm = np.asarray(
                solve_fixed_q7_tcp_pose(
                    tuple(float(value) for value in command_arm),
                    tuple(float(value) for value in command_tcp[:3, 3]),
                    target_rotation=command_tcp[:3, :3],
                    maximum_iterations=8,
                )
            )
            sample = observe_loaded("mixed_grip_physical_insert_plane_guarded", command_arm)
            wrench = np.asarray(sample["observed"]["compensated_wrench_task"])
            values = (
                abs(float(wrench[2])),
                float(np.linalg.norm(wrench[:2])),
                float(np.linalg.norm(wrench[3:5])),
                abs(float(wrench[5])),
            )
            limits = (
                float(hard["hard_axial_force_n"]),
                float(hard["hard_lateral_force_n"]),
                float(hard["hard_bending_moment_nm"]),
                float(hard["hard_torsional_moment_nm"]),
            )
            if any(value > limit for value, limit in zip(values, limits)):
                raise RuntimeError("HARD_GATE_DURING_FACE_APPROACH")
            if max(abs(float(wrench[2])), float(np.linalg.norm(wrench[:3]))) >= contact_threshold:
                face_contact = sample
                break
        if face_contact is None:
            raise RuntimeError("FACE_CONTACT_NOT_OBSERVED")
        hold_wrenches = []
        for _ in range(max(1, int(round(float(controller_config["active_probe"]["contact_hold_s"]) * rate)))):
            sample = observe_loaded("mixed_grip_physical_insert_plane_contact_hold", command_arm)
            hold_wrenches.append(np.asarray(sample["observed"]["compensated_wrench_task"]))
        mean_contact_wrench = np.mean(np.asarray(hold_wrenches), axis=0)
        contact_force = float(np.linalg.norm(mean_contact_wrench[:3]))
        face_contact_pass = contact_force >= float(gates["minimum_credible_contact_force_n"])
        report["stages"].append({
            "stage": "RECEPTACLE_RIM_PLANAR_CONTACT",
            "passed": face_contact_pass,
            "mean_contact_wrench_assembly": mean_contact_wrench.tolist(),
            "contact_force_magnitude_n": contact_force,
            "contact_detected_from": "hand2arm_wrist_wrench_only",
        })
        if not face_contact_pass:
            raise RuntimeError("WRIST_WRENCH_PATH_INVALID")

        # Unload, return to the exact nominal command, and refresh the
        # same-scene payload baseline before the nominal insertion.
        move_tcp(
            plane_lateral_tcp,
            max(1.0, float(np.linalg.norm(command_tcp[:3, 3] - plane_lateral_tcp[:3, 3])) / 0.001),
            "mixed_grip_preinsert_plane_contact_backoff",
            0.001875,
        )
        move_tcp(
            base_tcp,
            3.0,
            "mixed_grip_preinsert_plane_contact_recenter",
            0.001875,
        )
        recapture_raw = []
        for _ in range(wrist_config.payload_baseline_window_steps):
            sample = observe_loaded("unsupported_final_hold", base_arm)
            recapture_raw.append(sample["raw"])
        payload_baseline = monitor.capture_payload_baseline()
        report["wrench_baselines"]["post_contact_payload_baseline_canonical"] = payload_baseline
        report["wrench_baselines"]["post_contact_raw_mean"] = np.mean(np.asarray(recapture_raw), axis=0).tolist()

        # Now move most of the 12 mm nominal free-space gap.  Run 2 measured
        # no contact through 13.5 mm, so 11 mm remains a conservative
        # force-checked pre-guide command rather than a threshold relaxation.
        coarse_advance = float(nominal["coarse_free_space_advance_m"])
        coarse_speed = float(nominal["coarse_approach_speed_m_s"])
        coarse_tcp = base_tcp.copy()
        coarse_tcp[:3, 3] += initial_axis * coarse_advance
        coarse_samples = move_tcp(
            coarse_tcp,
            coarse_advance / coarse_speed,
            "mixed_grip_preinsert_coarse_free_space",
            1.875 * coarse_speed,
        )
        maximum_coarse_force = max(
            float(np.linalg.norm(np.asarray(item["observed"]["compensated_wrench_task"])[:3]))
            for item in coarse_samples
        )
        if maximum_coarse_force > contact_threshold:
            raise RuntimeError("UNEXPECTED_CONTACT_DURING_NOMINAL_COARSE_APPROACH")

        # Enter the requested depth stage from a centered, no-contact nominal
        # pose.  Stage completion is based on robot TCP FK.  Object poses and
        # contact-pair identities remain post-hoc diagnostics only.
        controller_state = ControllerState(phase=InsertionState.GUARDED_APPROACH)
        controller_origin_tcp = assembly_tcp_from_grasp_tcp(
            initial_tcp_measured, grasp_to_assembly
        )
        controller_command_tcp = command_tcp.copy()
        nominal_path_start = diagnostics["total_absolute_tcp_path_m"]
        controller_status_counts = {}
        contact_class_counts = {}
        correction_metrics = {
            "maximum_commanded_xy_speed_m_s": 0.0,
            "maximum_commanded_tilt_speed_rad_s": 0.0,
            "absolute_xy_correction_path_m": 0.0,
            "absolute_tilt_correction_path_rad": 0.0,
            "postcontact_absolute_xy_correction_path_m": 0.0,
            "postcontact_absolute_tilt_correction_path_rad": 0.0,
            "nonzero_xy_command_steps": 0,
            "nonzero_tilt_command_steps": 0,
            "first_credible_contact_step": None,
            "first_credible_contact_wrench": None,
            "first_credible_contact_score_n": None,
            "minimum_postcontact_score_n": None,
        }
        peak_wrench = np.zeros(4, dtype=np.float64)
        last_sample = coarse_samples[-1]
        guide_shell_front_from_body_m = -0.5 * proxy.plug.nose_chamfer_length
        guide_overlap_target = float(document["experiment"]["guide_entry_margin_m"]) + 0.00005
        guide_body_depth = (
            proxy.receptacle.entrance_chamfer_length
            - guide_shell_front_from_body_m
            + guide_overlap_target
        )
        target_body_depths = {
            "depth_3": 0.003,
            "guide_0p25": guide_body_depth,
            "additional_1": guide_body_depth + 0.001,
            "depth_6": 0.006,
            "full_9": proxy.insertion_depth,
        }
        target_body_depth = float(target_body_depths[arguments.target_stage])
        planned_target_progress = proxy.preinsert_gap + target_body_depth
        checkpoint_depths = tuple(
            sorted(
                (
                    ("DEPTH_3MM", 0.003),
                    ("GUIDE_OVERLAP_0P25MM", guide_body_depth),
                    ("ADDITIONAL_DEPTH_1MM", guide_body_depth + 0.001),
                    ("DEPTH_6MM", 0.006),
                    ("FULL_CONFIGURED_9MM", proxy.insertion_depth),
                ),
                key=lambda item: item[1],
            )
        )
        checkpoint_results = []
        checkpoint_names = set()
        maximum_steps = int(round(float(controller_config["motion"]["maximum_duration_s"]) * rate))
        nominal_terminal = "TIMEOUT"
        allowed_nominal_path = (
            planned_target_progress
            - coarse_advance
            + float(gates["maximum_nominal_path_margin_m"])
        )
        previous_rigid_twist = np.zeros(6, dtype=np.float64)
        request_soft_backoff = False
        response_identified = False

        def identify_contact_response(center_tcp):
            nonlocal command_tcp, command_arm
            specification = document["contact_response_identification"]
            translation_amplitude = float(
                specification["translation_amplitude_m"]
            )
            rotation_amplitude = float(specification["rotation_amplitude_rad"])
            move_duration = float(specification["move_duration_s"])
            hold_steps = max(
                1, int(round(float(specification["hold_duration_s"]) * rate))
            )
            axes = (
                ("X", 0, translation_amplitude),
                ("Y", 1, translation_amplitude),
                ("Rx", 3, rotation_amplitude),
                ("Ry", 4, rotation_amplitude),
            )
            center = np.asarray(center_tcp, dtype=np.float64).copy()
            command_tcp = center.copy()
            cases = []

            def pose_for(axis_index, signed_amplitude):
                target = center.copy()
                if axis_index < 3:
                    target[:3, 3] += base_tcp[:3, axis_index] * signed_amplitude
                else:
                    center_assembly = assembly_tcp_from_grasp_tcp(
                        center, grasp_to_assembly
                    )
                    vector = np.zeros(3, dtype=np.float64)
                    vector[axis_index - 3] = signed_amplitude
                    target[:3, :3] = (
                        _exp_rotation(base_tcp[:3, :3] @ vector)
                        @ center[:3, :3]
                    )
                    target[:3, 3] = (
                        center_assembly[:3, 3]
                        - target[:3, :3] @ grasp_to_assembly
                    )
                return target

            def held_wrench(label):
                values = []
                for _ in range(hold_steps):
                    sample = observe_loaded(label, command_arm)
                    wrench_value = np.asarray(
                        sample["observed"]["compensated_wrench_task"],
                        dtype=np.float64,
                    )
                    scalars = np.asarray(
                        (
                            abs(wrench_value[2]),
                            np.linalg.norm(wrench_value[:2]),
                            np.linalg.norm(wrench_value[3:5]),
                            abs(wrench_value[5]),
                        )
                    )
                    hard_values = np.asarray(
                        (
                            float(hard["hard_axial_force_n"]),
                            float(hard["hard_lateral_force_n"]),
                            float(hard["hard_bending_moment_nm"]),
                            float(hard["hard_torsional_moment_nm"]),
                        )
                    )
                    if np.any(scalars > 0.70 * hard_values):
                        raise RuntimeError(
                            "CONTACT_RESPONSE_IDENTIFICATION_SOFT_GATE"
                        )
                    values.append(wrench_value)
                return np.mean(np.asarray(values), axis=0)

            baseline = held_wrench("contact_response_center_baseline")
            columns = []
            for name, axis_index, amplitude in axes:
                signed = {}
                for sign_name, sign in (("plus", 1.0), ("minus", -1.0)):
                    target = pose_for(axis_index, sign * amplitude)
                    move_tcp(
                        target,
                        move_duration,
                        f"contact_response_{sign_name}_{name}",
                        max(
                            translation_amplitude / move_duration,
                            0.02
                            * rotation_amplitude
                            / move_duration,
                        ),
                    )
                    signed[sign_name] = held_wrench(
                        f"contact_response_{sign_name}_{name}_hold"
                    )
                    move_tcp(
                        center,
                        move_duration,
                        f"contact_response_return_{name}_{sign_name}",
                        max(
                            translation_amplitude / move_duration,
                            0.02
                            * rotation_amplitude
                            / move_duration,
                        ),
                    )
                    held_wrench(f"contact_response_return_{name}_{sign_name}_hold")
                column = (signed["plus"] - signed["minus"]) / (2.0 * amplitude)
                columns.append(column)
                cases.append(
                    {
                        "motion_axis": name,
                        "amplitude": amplitude,
                        "plus_mean_wrench": signed["plus"].tolist(),
                        "minus_mean_wrench": signed["minus"].tolist(),
                        "central_difference_column": column.tolist(),
                    }
                )
            matrix = np.column_stack(columns)
            scaled_matrix = matrix.copy()
            scaled_matrix[:, 2:] /= float(
                specification["rotation_effective_length_m"]
            )
            singular_values = np.linalg.svd(
                scaled_matrix, compute_uv=False
            )
            condition = float(
                singular_values[0] / singular_values[-1]
                if singular_values[-1] > 1.0e-12
                else math.inf
            )
            return {
                "stage": "CONTACT_RESPONSE_IDENTIFICATION",
                "controller_inputs": [
                    "robot_joint_state",
                    "robot_tcp_fk",
                    "hand2arm_wrist_wrench",
                    "controller_history",
                ],
                "forbidden_inputs_used": [],
                "baseline_wrench": baseline.tolist(),
                "cases": cases,
                "response_matrix_wrench_per_motion": matrix.tolist(),
                "scaled_singular_values": singular_values.tolist(),
                "scaled_condition_number": condition,
                "passed": bool(np.all(np.isfinite(matrix))),
            }
        insertion_attempt = 0
        nominal_terminal = None
        request_soft_backoff = False
        c2_branch_retry = {"attempted": False, "rotated_rad": 0.0, "reason": None}
        while True:
            for _ in range(maximum_steps):
                measured_tcp = last_sample["tcp"]
                measured_assembly_tcp = assembly_tcp_from_grasp_tcp(
                    measured_tcp, grasp_to_assembly
                )
                relative = base_tcp[:3, :3].T @ (
                    measured_assembly_tcp[:3, 3]
                    - controller_origin_tcp[:3, 3]
                )
                relative_rotation = (
                    controller_origin_tcp[:3, :3].T
                    @ measured_assembly_tcp[:3, :3]
                )
                wrench = np.asarray(last_sample["observed"]["compensated_wrench_task"])
                if arguments.control_mode == "compliant":
                    observation = InsertionObservation(
                        timestamp_s=global_step * dt,
                        sample_age_s=0.0,
                        wrench_assembly=tuple(float(value) for value in wrench),
                        tcp_position_assembly_m=tuple(float(value) for value in relative),
                        tcp_rotation_vector_assembly_rad=tuple(
                            float(value)
                            for value in _xyz_from_rotation(relative_rotation)
                        ),
                        vision_control_authorized=True,
                        synchronized_capture=True,
                        ft_valid=True,
                        ft_tared=True,
                        payload_compensated=True,
                    )
                    action = step_compliant_insertion(
                        controller_config, controller_state, observation
                    )
                    controller_state = action.next_state
                    status = action.status
                    twist = np.asarray(action.twist_assembly, dtype=np.float64)
                    contact_name = action.contact_class.value
                else:
                    target_twist = np.zeros(6, dtype=np.float64)
                    target_twist[2] = guard_speed
                    linear_delta = float(
                        controller_config["motion"][
                            "maximum_linear_acceleration_m_s2"
                        ]
                    ) * dt
                    twist = previous_rigid_twist + np.clip(
                        target_twist - previous_rigid_twist,
                        -linear_delta,
                        linear_delta,
                    )
                    previous_rigid_twist = twist.copy()
                    status = "RIGID_TCP_SERVO"
                    contact_name = "NOT_CLASSIFIED_RIGID"
                controller_status_counts[status] = (
                    controller_status_counts.get(status, 0) + 1
                )
                contact_class_counts[contact_name] = (
                    contact_class_counts.get(contact_name, 0) + 1
                )
                xy_speed = float(np.linalg.norm(twist[:2]))
                tilt_speed = float(np.linalg.norm(twist[3:5]))
                correction_metrics["maximum_commanded_xy_speed_m_s"] = max(
                    correction_metrics["maximum_commanded_xy_speed_m_s"], xy_speed
                )
                correction_metrics["maximum_commanded_tilt_speed_rad_s"] = max(
                    correction_metrics["maximum_commanded_tilt_speed_rad_s"],
                    tilt_speed,
                )
                correction_metrics["absolute_xy_correction_path_m"] += xy_speed * dt
                correction_metrics["absolute_tilt_correction_path_rad"] += (
                    tilt_speed * dt
                )
                if correction_metrics["first_credible_contact_step"] is not None:
                    correction_metrics[
                        "postcontact_absolute_xy_correction_path_m"
                    ] += xy_speed * dt
                    correction_metrics[
                        "postcontact_absolute_tilt_correction_path_rad"
                    ] += tilt_speed * dt
                correction_metrics["nonzero_xy_command_steps"] += int(
                    xy_speed > float(
                        controller_config["motion"][
                            "minimum_effective_linear_speed_m_s"
                        ]
                    )
                )
                correction_metrics["nonzero_tilt_command_steps"] += int(
                    tilt_speed > 1.0e-6
                )
                controller_command_tcp = integrate_assembly_twist_on_grasp_tcp(
                    controller_command_tcp,
                    twist,
                    base_tcp[:3, :3],
                    grasp_to_assembly,
                    dt,
                )
                command_arm = np.asarray(
                    solve_fixed_q7_tcp_pose(
                        tuple(float(value) for value in command_arm),
                        tuple(float(value) for value in controller_command_tcp[:3, 3]),
                        target_rotation=controller_command_tcp[:3, :3],
                        maximum_iterations=8,
                    )
                )
                last_sample = observe_loaded("mixed_grip_physical_insert_nominal", command_arm)
                wrench = np.asarray(last_sample["observed"]["compensated_wrench_task"])
                contact_score = float(
                    abs(wrench[2])
                    + float(
                        controller_config["active_probe"][
                            "bending_weight_n_per_nm"
                        ]
                    )
                    * np.linalg.norm(wrench[3:5])
                )
                credible_contact = bool(
                    abs(wrench[2])
                    >= float(
                        controller_config["contact_classifier"]["axial_contact_n"]
                    )
                    or np.linalg.norm(wrench[:2])
                    >= float(
                        controller_config["contact_classifier"]["lateral_contact_n"]
                    )
                    or np.linalg.norm(wrench[3:5])
                    >= float(
                        controller_config["contact_classifier"][
                            "bending_contact_nm"
                        ]
                    )
                )
                if (
                    credible_contact
                    and correction_metrics["first_credible_contact_step"] is None
                ):
                    correction_metrics["first_credible_contact_step"] = global_step
                    correction_metrics["first_credible_contact_wrench"] = (
                        wrench.tolist()
                    )
                    correction_metrics["first_credible_contact_score_n"] = (
                        contact_score
                    )
                    correction_metrics["minimum_postcontact_score_n"] = contact_score
                    if arguments.identify_contact_response:
                        report["contact_response_identification"] = (
                            identify_contact_response(controller_command_tcp)
                        )
                        response_identified = True
                        report["nominal_physics_valid"] = None
                        report["full_nominal_seated"] = False
                        report["bcapture_scan_run"] = False
                        report["bcapture_lock_reason"] = (
                            "CONTACT_RESPONSE_IDENTIFICATION_ONLY"
                        )
                        report["passed"] = report[
                            "contact_response_identification"
                        ]["passed"]
                        passed = report["passed"]
                        raise _ResponseIdentificationComplete()
                elif correction_metrics["first_credible_contact_step"] is not None:
                    correction_metrics["minimum_postcontact_score_n"] = min(
                        correction_metrics["minimum_postcontact_score_n"],
                        contact_score,
                    )
                peak_wrench = np.maximum(
                    peak_wrench,
                    np.asarray((abs(wrench[2]), np.linalg.norm(wrench[:2]), np.linalg.norm(wrench[3:5]), abs(wrench[5]))),
                )
                raw_hard_now = bool(
                    peak_wrench[0] > float(hard["hard_axial_force_n"])
                    or peak_wrench[1] > float(hard["hard_lateral_force_n"])
                    or peak_wrench[2] > float(hard["hard_bending_moment_nm"])
                    or peak_wrench[3] > float(hard["hard_torsional_moment_nm"])
                )
                if raw_hard_now:
                    nominal_terminal = "RAW_HARD_SAFETY_GATE"
                    break
                last_assembly_tcp = assembly_tcp_from_grasp_tcp(
                    last_sample["tcp"], grasp_to_assembly
                )
                initial_assembly_tcp = assembly_tcp_from_grasp_tcp(
                    initial_tcp_measured, grasp_to_assembly
                )
                measured_from_preinsert = float(
                    (
                        last_assembly_tcp[:3, 3]
                        - initial_assembly_tcp[:3, 3]
                    )
                    @ initial_axis
                )
                estimated_body_depth_from_fk = measured_from_preinsert - proxy.preinsert_gap
                for checkpoint_name, checkpoint_depth in checkpoint_depths:
                    if (
                        checkpoint_name not in checkpoint_names
                        and estimated_body_depth_from_fk >= checkpoint_depth
                    ):
                        checkpoint_names.add(checkpoint_name)
                        checkpoint_results.append(
                            {
                                "stage": checkpoint_name,
                                "planned_body_depth_m": checkpoint_depth,
                                "actual_fk_body_depth_m": estimated_body_depth_from_fk,
                                "max_axial_force_n": float(peak_wrench[0]),
                                "max_lateral_force_n": float(peak_wrench[1]),
                                "max_bending_moment_nm": float(peak_wrench[2]),
                                "max_torsional_moment_nm": float(peak_wrench[3]),
                                "max_tcp_step_m": diagnostics["maximum_tcp_step_m"],
                                "total_path_m": diagnostics[
                                    "total_absolute_tcp_path_m"
                                ]
                                - nominal_path_start,
                                "control_inputs": [
                                    "robot_joint_state",
                                    "robot_tcp_fk",
                                    "hand2arm_wrist_wrench",
                                    "controller_history",
                                ],
                            }
                        )
                nominal_path = diagnostics["total_absolute_tcp_path_m"] - nominal_path_start
                allowed_step = max(
                    float(gates["expected_motion_multiplier"]) * max(float(np.linalg.norm(twist[:3])), guard_speed) * dt,
                    float(gates["minimum_max_step_m"]),
                )
                if last_sample["record"]["tcp_step_m"] > allowed_step:
                    nominal_terminal = "PHYSICS_INVALID_TCP_STEP"
                    break
                if float(np.linalg.norm(last_sample["tcp"][:3, 3] - initial_tcp_measured[:3, 3])) > float(gates["absolute_physics_invalid_motion_m"]):
                    nominal_terminal = "PHYSICS_INVALID_OVER_50MM"
                    break
                # Plug/Nut/Receptacle state is recorded by a non-controlling
                # evidence side channel.  Constraint drift, velocity and energy
                # are scored only after the controller has terminated; they do
                # not select a twist, correction direction or terminal state.
                measured_tilt = float(
                    np.linalg.norm(_xyz_from_rotation(relative_rotation)[:2])
                )
                if measured_tilt > float(
                    controller_config["motion"]["maximum_search_angle_rad"]
                ):
                    nominal_terminal = "TILT_SEARCH_BOUND_EXCEEDED"
                    break
                planned_nominal_path = planned_target_progress - coarse_advance
                active_probe_config = controller_config["active_probe"]
                per_realign_path_budget = (
                    8.0
                    * float(active_probe_config["maximum_leg_duration_s"])
                    * float(active_probe_config["speed_m_s"])
                    # A bounded recovery first unloads along -Z and then returns
                    # to the contact depth along +Z.  Account for both legs;
                    # counting only the unload leg falsely invalidates otherwise
                    # bounded guide-exit recovery as excessive TCP travel.
                    + 2.0
                    * max(
                        float(active_probe_config["contact_unload_distance_m"]),
                        float(
                            active_probe_config[
                                "angular_contact_unload_distance_m"
                            ]
                        ),
                    )
                    + float(active_probe_config["unloaded_centering_distance_m"])
                )
                recovery_phase_active = controller_state.phase in {
                    InsertionState.ACTIVE_PROBE,
                    InsertionState.CONTACT_UNLOAD,
                    InsertionState.UNLOADED_CENTERING,
                }
                budgeted_realign_cycles = controller_state.contact_realign_count + int(
                    recovery_phase_active
                )
                allowed_nominal_path = (
                    planned_nominal_path
                    + float(gates["maximum_nominal_path_margin_m"])
                    + budgeted_realign_cycles * per_realign_path_budget
                )
                if nominal_path > allowed_nominal_path:
                    nominal_terminal = "NOMINAL_PATH_EXCEEDED"
                    break
                if (
                    arguments.control_mode == "compliant"
                    and controller_state.phase is InsertionState.SAFE_ABORT
                ):
                    nominal_terminal = controller_state.abort_reason or "SAFE_ABORT"
                    break
                soft_values = 0.70 * np.asarray(
                    (
                        float(hard["hard_axial_force_n"]),
                        float(hard["hard_lateral_force_n"]),
                        float(hard["hard_bending_moment_nm"]),
                        float(hard["hard_torsional_moment_nm"]),
                    )
                )
                if np.any(
                    np.asarray(
                        (
                            abs(wrench[2]),
                            np.linalg.norm(wrench[:2]),
                            np.linalg.norm(wrench[3:5]),
                            abs(wrench[5]),
                        )
                    )
                    > soft_values
                ):
                    nominal_terminal = "SOFT_GATE_BACKOFF_STOP"
                    request_soft_backoff = True
                    break
                if (
                    arguments.control_mode == "compliant"
                    and controller_state.phase is InsertionState.BACKOFF
                ):
                    nominal_terminal = "CONTROLLER_BACKOFF_STOP"
                    request_soft_backoff = True
                    break
                if measured_from_preinsert >= planned_target_progress:
                    nominal_terminal = "PLANNED_DEPTH_REACHED"
                    break

            if (
                request_soft_backoff
                and insertion_attempt == 0
                and arguments.control_mode == "compliant"
                and measured_from_preinsert < 0.5 * planned_target_progress
            ):
                backoff_speed = float(controller_config["recovery"]["backoff_speed_m_s"])
                backoff_distance = float(controller_config["recovery"]["backoff_distance_m"])
                backoff_steps = max(1, int(math.ceil(backoff_distance / (backoff_speed * dt))))
                for _ in range(backoff_steps):
                    controller_command_tcp[:3, 3] -= initial_axis * backoff_speed * dt
                    command_arm = np.asarray(
                        solve_fixed_q7_tcp_pose(
                            tuple(float(value) for value in command_arm),
                            tuple(float(value) for value in controller_command_tcp[:3, 3]),
                            target_rotation=controller_command_tcp[:3, :3],
                            maximum_iterations=8,
                        )
                    )
                    last_sample = observe_loaded(
                        "mixed_grip_physical_insert_branch_retry_retract",
                        command_arm,
                    )
                axis = initial_axis / np.linalg.norm(initial_axis)
                k = np.array(
                    [
                        [0.0, -axis[2], axis[1]],
                        [axis[2], 0.0, -axis[0]],
                        [-axis[1], axis[0], 0.0],
                    ]
                )
                flip = np.eye(3) + 2.0 * (k @ k)
                controller_command_tcp[:3, :3] = (
                    flip @ controller_command_tcp[:3, :3]
                )
                # Select the other C2 branch with the wrist roll joint
                # (iiwa_joint_7): rotating the TCP 180 degrees about the
                # assembly axis is exactly a joint-7 roll.  The local IK keeps
                # q7 fixed, so presetting it makes the flipped pose reachable.
                command_arm[6] = command_arm[6] + math.pi
                if command_arm[6] > 3.0543:
                    command_arm[6] -= 2.0 * math.pi
                if abs(command_arm[6]) > 3.0543:
                    raise RuntimeError(
                        "C2 branch flip unreachable within joint 7 hard limits"
                    )
                controller_state = ControllerState(
                    phase=InsertionState.GUARDED_APPROACH
                )
                previous_rigid_twist = np.zeros(6, dtype=np.float64)
                insertion_attempt += 1
                c2_branch_retry = {
                    "attempted": True,
                    "rotated_rad": float(math.pi),
                    "reason": nominal_terminal,
                }
                nominal_terminal = None
                request_soft_backoff = False
                continue
            break

        terminal_sample = last_sample
        if request_soft_backoff:
            backoff_speed = float(controller_config["recovery"]["backoff_speed_m_s"])
            backoff_distance = float(controller_config["recovery"]["backoff_distance_m"])
            backoff_steps = max(1, int(math.ceil(backoff_distance / (backoff_speed * dt))))
            for _ in range(backoff_steps):
                controller_command_tcp[:3, 3] -= initial_axis * backoff_speed * dt
                command_arm = np.asarray(
                    solve_fixed_q7_tcp_pose(
                        tuple(float(value) for value in command_arm),
                        tuple(float(value) for value in controller_command_tcp[:3, 3]),
                        target_rotation=controller_command_tcp[:3, :3],
                        maximum_iterations=8,
                    )
                )
                last_sample = observe_loaded(
                    "mixed_grip_physical_insert_soft_gate_backoff", command_arm
                )

        insertion_terminal_sample = last_sample
        screw_result = {"status": "NOT_REQUESTED"}
        if (
            arguments.screw_after_seat
            and not request_soft_backoff
            and nominal_terminal == "PLANNED_DEPTH_REACHED"
        ):
            screw_segment_rad = math.radians(150.0)
            screw_steps = 600
            q7_before = float(command_arm[6])
            screw_trip = None
            hard_axial = float(hard["hard_axial_force_n"])
            hard_lateral = float(hard["hard_lateral_force_n"])
            hard_bend = float(hard["hard_bending_moment_nm"])
            hard_torsion = float(hard["hard_torsional_moment_nm"])
            screw_soft = 0.70 * np.asarray(
                (hard_axial, hard_lateral, hard_bend, hard_torsion)
            )
            baseline_wrench = np.asarray(
                insertion_terminal_sample["observed"][
                    "compensated_wrench_task"
                ],
                dtype=np.float64,
            )
            for step in range(screw_steps):
                command_arm[6] = q7_before + screw_segment_rad * (step + 1) / screw_steps
                screw_sample = observe_loaded(
                    "mixed_grip_physical_insert_screw_segment", command_arm
                )
                wrench = np.asarray(
                    screw_sample["observed"]["compensated_wrench_task"],
                    dtype=np.float64,
                )
                # gate on the wrench DELTA from the seated baseline: the
                # axial seating preload is expected and must not trip the
                # screw gates; torsional/lateral excursions must stay bounded
                delta_wrench = wrench - baseline_wrench
                delta_magnitudes = np.asarray(
                    (
                        abs(delta_wrench[2]),
                        np.linalg.norm(delta_wrench[:2]),
                        np.linalg.norm(delta_wrench[3:5]),
                        abs(delta_wrench[5]),
                    )
                )
                if np.any(delta_magnitudes > screw_soft):
                    screw_trip = f"SCREW_SOFT_GATE_AT_STEP_{step}"
                    break
            screw_result = {
                "status": "COMPLETED" if screw_trip is None else screw_trip,
                "total_rotation_rad": (
                    screw_segment_rad if screw_trip is None else
                    screw_segment_rad * (step + 1) / screw_steps
                ),
                "steps": screw_steps,
                "q7_before_rad": q7_before,
                "q7_after_rad": float(command_arm[6]),
                "anti_slip": "GRIP_HELD_POSTHOC_PENDING",
            }
            last_sample = insertion_terminal_sample

        nominal_path = diagnostics["total_absolute_tcp_path_m"] - nominal_path_start
        posthoc_body_progress = float((terminal_sample["body_pose"][0] - initial_body_pose[0]) @ initial_axis)
        receptacle_position, receptacle_rotation = terminal_sample["receptacle_pose"]
        body_position, body_rotation = terminal_sample["body_pose"]
        body_in_receptacle = receptacle_rotation.T @ (body_position - receptacle_position)
        posthoc_lateral = float(np.linalg.norm(body_in_receptacle[:2]))
        posthoc_axis_error = math.acos(
            max(
                -1.0,
                min(
                    1.0,
                    float(body_rotation[:, 2] @ receptacle_rotation[:, 2]),
                ),
            )
        )
        guide_overlap = float(
            body_in_receptacle[2]
            + guide_shell_front_from_body_m
            - proxy.receptacle.entrance_chamfer_length
        )
        maximum_effective_lateral = (
            proxy.radial_clearance + proxy.physics.contact_offset_m
        )
        posthoc_effective_lateral = effective_lateral_posthoc(
            posthoc_lateral,
            posthoc_axis_error,
            proxy.plug.guide_length,
        )
        posthoc_guided = bool(
            guide_overlap >= float(document["experiment"]["guide_entry_margin_m"])
            and posthoc_effective_lateral <= maximum_effective_lateral
        )
        hard_gate = bool(
            peak_wrench[0] > float(hard["hard_axial_force_n"])
            or peak_wrench[1] > float(hard["hard_lateral_force_n"])
            or peak_wrench[2] > float(hard["hard_bending_moment_nm"])
            or peak_wrench[3] > float(hard["hard_torsional_moment_nm"])
        )
        posthoc_target_depth_reached = bool(
            body_in_receptacle[2] >= target_body_depth - 0.00010
        )
        posthoc_depth_only_seated = bool(
            body_in_receptacle[2] >= proxy.insertion_depth - 0.00010
        )
        # A Z coordinate alone cannot establish a seated connector.  In
        # particular, a tilted proxy can cross the nominal depth plane while
        # remaining outside the validated guide pose.  Keep the depth-only
        # observation for diagnostics, but fail the formal seated gate unless
        # the body is also in the posthoc guide envelope.
        posthoc_seated = full_seated_posthoc(
            float(body_in_receptacle[2]),
            float(proxy.insertion_depth),
            posthoc_guided,
        )
        physics_integrity_valid = bool(
            nominal_terminal == "PLANNED_DEPTH_REACHED"
            and posthoc_target_depth_reached
            and not hard_gate
            and diagnostics["maximum_tcp_step_m"] <= float(gates["minimum_max_step_m"])
            and diagnostics["maximum_plug_hand_translation_drift_m"] <= float(gates["maximum_plug_hand_translation_drift_m"])
            and diagnostics["maximum_plug_hand_rotation_drift_rad"] <= float(gates["maximum_plug_hand_rotation_drift_rad"])
            and diagnostics["maximum_nut_plug_translation_drift_m"] <= float(gates["maximum_plug_hand_translation_drift_m"])
            and diagnostics["maximum_plug_linear_velocity_m_s"] <= float(gates["maximum_linear_velocity_m_s"])
            and diagnostics["maximum_plug_angular_velocity_rad_s"] <= float(gates["maximum_angular_velocity_rad_s"])
            and diagnostics["maximum_payload_kinetic_energy_j"] <= float(gates["maximum_kinetic_energy_j"])
        )
        formal_profile = arguments.collision_profile == "full"
        physics_valid = bool(physics_integrity_valid and formal_profile)
        full_nominal_seated = bool(
            physics_valid
            and arguments.target_stage == "full_9"
            and posthoc_seated
        )
        stage_passed = bool(
            physics_valid
            and (
                arguments.target_stage != "full_9"
                or full_nominal_seated
            )
        )
        reported_terminal = nominal_terminal
        report["c2_yaw_branch_retry"] = dict(c2_branch_retry)
        report["screw_after_seat"] = dict(screw_result)
        if (
            nominal_terminal == "PLANNED_DEPTH_REACHED"
            and arguments.target_stage == "full_9"
            and posthoc_depth_only_seated
            and not posthoc_guided
        ):
            reported_terminal = "DEPTH_PLANE_REACHED_POSTHOC_NOT_GUIDED"
        meaningful_xy = bool(
            correction_metrics["postcontact_absolute_xy_correction_path_m"]
            >= float(document["experiment"]["meaningful_xy_correction_m"])
        )
        meaningful_tilt = bool(
            correction_metrics[
                "postcontact_absolute_tilt_correction_path_rad"
            ]
            >= float(document["experiment"]["meaningful_tilt_correction_rad"])
        )
        commanded_assembly_tcp = assembly_tcp_from_grasp_tcp(
            controller_command_tcp, grasp_to_assembly
        )
        coarse_assembly_tcp = assembly_tcp_from_grasp_tcp(
            coarse_tcp, grasp_to_assembly
        )
        commanded_relative_translation = base_tcp[:3, :3].T @ (
            commanded_assembly_tcp[:3, 3] - coarse_assembly_tcp[:3, 3]
        )
        commanded_relative_rotation = _xyz_from_rotation(
            base_tcp[:3, :3].T @ controller_command_tcp[:3, :3]
        )
        credible_contact_formed = (
            correction_metrics["first_credible_contact_step"] is not None
        )
        score_drop = 0.0
        if credible_contact_formed:
            score_drop = float(
                correction_metrics["first_credible_contact_score_n"]
                - correction_metrics["minimum_postcontact_score_n"]
            )
        load_reduced_after_contact = bool(
            score_drop
            >= float(document["experiment"]["minimum_contact_score_drop_n"])
        )
        if full_nominal_seated and not credible_contact_formed:
            capture_class = "BFREE"
        elif (
            full_nominal_seated
            and credible_contact_formed
            and (meaningful_xy or meaningful_tilt)
            and load_reduced_after_contact
        ):
            capture_class = "BCONTACT_CAPTURE"
        else:
            capture_class = "BSAFE_FAIL"
        correction_metrics.update(
            {
                "meaningful_xy_correction": meaningful_xy,
                "meaningful_tilt_correction": meaningful_tilt,
                "credible_contact_formed": credible_contact_formed,
                "postcontact_score_drop_n": score_drop,
                "load_reduced_after_contact": load_reduced_after_contact,
                "net_commanded_translation_from_coarse_m": (
                    commanded_relative_translation.tolist()
                ),
                "net_commanded_rotation_from_coarse_rad": (
                    commanded_relative_rotation.tolist()
                ),
            }
        )
        nominal_result = {
            "stage": "V2_ORIGIN_NOMINAL_STAGED_INSERTION",
            "passed": stage_passed,
            "physics_integrity_valid": physics_integrity_valid,
            "formal_profile": formal_profile,
            "terminal_state": reported_terminal,
            "controller_status_counts": controller_status_counts,
            "contact_class_counts": contact_class_counts,
            "controller_final_probe": {
                "direction_order_indices": list(
                    controller_state.probe_direction_order
                ),
                "relative_endpoint_scores_n": list(
                    float(value) if math.isfinite(value) else None
                    for value in controller_state.probe_scores
                ),
                "selected_xy": list(controller_state.probe_selected_xy),
                "selected_tilt": list(
                    controller_state.probe_selected_tilt
                ),
                "retained_tilt": list(
                    controller_state.retained_probe_tilt
                ),
                "retained_xy": list(
                    controller_state.retained_probe_xy
                ),
                "cumulative_unloaded_tilt_rad": float(
                    controller_state.cumulative_unloaded_tilt_rad
                ),
                "selection_history_xy_rx_ry": [
                    list(values)
                    for values in controller_state.probe_selection_history
                ],
                "probe_total_steps": controller_state.probe_total_steps,
            },
            "correction_metrics": correction_metrics,
            "capture_class": capture_class,
            "planned_target_stage": arguments.target_stage,
            "planned_target_body_depth_m": target_body_depth,
            "planned_progress_from_preinsert_m": planned_target_progress,
            "planned_nominal_path_m": planned_target_progress - coarse_advance,
            "depth_checkpoints": checkpoint_results,
            "posthoc_body_progress_m": posthoc_body_progress,
            "posthoc_body_origin_in_receptacle_m": body_in_receptacle.tolist(),
            "posthoc_guide_shell_front_from_body_m": guide_shell_front_from_body_m,
            "posthoc_guide_overlap_m": guide_overlap,
            "posthoc_lateral_error_m": posthoc_lateral,
            "posthoc_effective_lateral_m": posthoc_effective_lateral,
            "posthoc_effective_lateral_formula": (
                "lateral + plug_guide_length * tan(axis_error)"
            ),
            "posthoc_maximum_effective_lateral_m": maximum_effective_lateral,
            "posthoc_axis_error_rad": posthoc_axis_error,
            "posthoc_physically_guided_entry": posthoc_guided,
            "posthoc_target_depth_reached": posthoc_target_depth_reached,
            "posthoc_depth_only_seated_plane_reached": (
                posthoc_depth_only_seated
            ),
            "posthoc_seated": posthoc_seated,
            "full_nominal_seated": full_nominal_seated,
            "nominal_total_absolute_path_m": nominal_path,
            "nominal_allowed_absolute_path_m": allowed_nominal_path,
            "contact_realign_count": controller_state.contact_realign_count,
            "peak_axial_force_n": float(peak_wrench[0]),
            "peak_lateral_force_n": float(peak_wrench[1]),
            "peak_bending_moment_nm": float(peak_wrench[2]),
            "peak_torsional_moment_nm": float(peak_wrench[3]),
            "hard_gate_triggered": hard_gate,
            "truth_used_by_controller": False,
            "truth_used_posthoc": True,
        }
        report["stages"].append(nominal_result)
        report["posthoc_first_contacts_by_phase"] = posthoc_first_contacts
        report["physics_diagnostics"] = diagnostics
        report["wrist_ft_monitor"] = monitor.report()
        report["nominal_physics_valid"] = physics_valid
        report["full_nominal_seated"] = full_nominal_seated
        report["bcapture_scan_run"] = False
        report["bcapture_lock_reason"] = "USER_LOCKED_THIS_ROUND"
        report["passed"] = stage_passed
        passed = stage_passed
        if not stage_passed:
            raise RuntimeError(f"NOMINAL_PHYSICS_FAILED:{nominal_terminal}")
    except _InhandValidationComplete:
        pass
    except _ResponseIdentificationComplete:
        pass
    except BaseException as exception:
        report["error"] = f"{type(exception).__name__}: {exception}"
        report["traceback"] = traceback.format_exc()
        if "bcapture_scan_run" not in report:
            report["bcapture_scan_run"] = False
            report["bcapture_lock_reason"] = "NOMINAL_PHYSICS_NOT_VALID"
        traceback.print_exc()
    finally:
        trace_stream.close()
        (output / "nominal_physics_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        stages = "\n".join(
            f"- {item['stage']}: {'PASSED' if item.get('passed') else 'FAILED'}"
            for item in report.get("stages", [])
        )
        (output / "REPORT.md").write_text(
            "# D38999 V2 nominal physical-validity ladder\n\n"
            f"Overall: `{'PASSED' if report.get('passed') else 'FAILED'}`\n\n"
            "This run uses the real iiwa/hand articulation and the hand2arm "
            "fixed-joint reaction wrench. The fixed grasp is explicitly "
            "`GRASP_LATCH_PROXY`. No Plug/Nut motion actuator is present.\n\n"
            "## Stair stages\n\n"
            + (stages or "- no stage completed")
            + "\n\nBcapture was not run in this command.\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "passed": report.get("passed", False),
                    "nominal_physics_valid": report.get("nominal_physics_valid", False),
                    "bcapture_scan_run": report.get("bcapture_scan_run", False),
                    "error": report.get("error"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        print(
            "ISAAC D38999 V2 NOMINAL PHYSICS "
            + ("PASSED" if passed else "FAILED"),
            flush=True,
        )
        app.close(exit_code=0 if passed else 1)


if __name__ == "__main__":
    main()
