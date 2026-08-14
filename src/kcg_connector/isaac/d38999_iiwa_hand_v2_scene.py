#!/usr/bin/env python3

"""Shared Isaac authoring for the real iiwa/hand/V2 load path.

This module deliberately contains no controller.  It authors one physical
tree before ``World.reset()`` and exposes the exact paths needed by the
direction-calibration and nominal-insertion runners.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from kcg_connector.postgrasp_error_injection import (
    PostGraspError,
    compose_nominal_with_error,
)


ROBOT_ROOT = "/World/HandArm"
ARTICULATION_ROOT = "/World/HandArm/Geometry/world"
HAND_BASE = (
    ARTICULATION_ROOT
    + "/iiwa_link_0/iiwa_link_1/iiwa_link_2/iiwa_link_3/iiwa_link_4"
    + "/iiwa_link_5/iiwa_link_6/iiwa_link_7/iiwa_link_ee/handbase_link"
)
GRASP_TCP = HAND_BASE + "/grasp_tcp"
HAND2ARM_JOINT = "/World/HandArm/Physics/hand2arm"
V2_ROOT = "/World/D38999InsertProxyV2"
RECEPTACLE = V2_ROOT + "/Receptacle"
PLUG = V2_ROOT + "/Plug"
PLUG_BODY = PLUG + "/Body"
COUPLING_NUT = PLUG + "/CouplingNut"
COUPLING_NUT_JOINT = PLUG + "/CouplingNutJoint"
GRASP_LATCH = "/World/V2NominalJoints/grasp_latch_proxy"
RECEPTACLE_WORLD_JOINT = (
    "/World/V2NominalJoints/receptacle_world_fixed"
)

COLLISION_PROFILES = (
    "full",
    "nose_guide_only",
    "no_coupling_nut",
    "no_c2_keys",
    "no_outer_nonmating",
)

ARM_NAMES = tuple(f"iiwa_joint_{index}" for index in range(1, 8))
HAND_NAMES = (
    "f1j1", "f1j2", "f1j3", "f2j1", "f2j2", "f3j1", "f3j2", "f3j3"
)


def _quat_wxyz_from_rotation(rotation):
    """Return a normalized scalar-first quaternion from a 3x3 matrix."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.allclose(
        matrix.T @ matrix, np.eye(3), atol=1.0e-8
    ):
        raise ValueError("rotation must be orthonormal")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = (
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(
                1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]
            ) * 2.0
            values = (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            )
        elif index == 1:
            scale = math.sqrt(
                1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]
            ) * 2.0
            values = (
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            )
        else:
            scale = math.sqrt(
                1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]
            ) * 2.0
            values = (
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            )
    result = np.asarray(values, dtype=np.float64)
    return result / np.linalg.norm(result)


def _rotation_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rotation_x = np.asarray(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))
    rotation_y = np.asarray(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
    rotation_z = np.asarray(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))
    return rotation_z @ rotation_y @ rotation_x


def _set_revolute_initial_state(stage, Sdf, joint_path, angle_rad):
    prim = stage.GetPrimAtPath(joint_path)
    if not prim.IsValid():
        raise RuntimeError(f"missing robot joint: {joint_path}")
    degrees = float(math.degrees(angle_rad))
    for name, value in (
        ("state:angular:physics:position", degrees),
        ("state:angular:physics:velocity", 0.0),
        ("drive:angular:physics:targetPosition", degrees),
        ("drive:angular:physics:targetVelocity", 0.0),
    ):
        attribute = prim.GetAttribute(name)
        if not attribute.IsValid():
            attribute = prim.CreateAttribute(name, Sdf.ValueTypeNames.Float)
        attribute.Set(value)


def author_robot_initial_state(stage, Sdf, arm_rad, open_hand_rad):
    arm = tuple(float(value) for value in arm_rad)
    hand = tuple(float(value) for value in open_hand_rad)
    if len(arm) != 7 or len(hand) != 8:
        raise ValueError("expected 7 arm and 8 hand joint positions")
    for name, angle in zip(ARM_NAMES, arm):
        _set_revolute_initial_state(
            stage, Sdf, f"/World/HandArm/Physics/{name}", angle
        )
    for name, angle in zip(HAND_NAMES, hand):
        _set_revolute_initial_state(
            stage, Sdf, f"/World/HandArm/Physics/{name}", angle
        )


def author_scene(
    *,
    stage,
    robot_asset: Path,
    v2_asset: Path,
    arm_rad,
    tcp_transform,
    initial_tcp_transform,
    proxy,
    add_reference_to_stage,
    Gf,
    Sdf,
    UsdGeom,
    UsdPhysics,
    include_payload: bool,
    latch_tcp_to_body_m: float,
    post_grasp_error: PostGraspError | None = None,
    preinsert_gap_m: float | None = None,
    receptacle_error_translation_assembly_m=(0.0, 0.0, 0.0),
    receptacle_error_rotation_xyz_rad=(0.0, 0.0, 0.0),
):
    """Author the complete load path, or an empty robot calibration scene."""

    add_reference_to_stage(str(robot_asset), ROBOT_ROOT)
    if not include_payload:
        return {
            "mode": "EMPTY_IIWA_HAND",
            "robot_root": ROBOT_ROOT,
            "articulation_root": ARTICULATION_ROOT,
            "hand2arm_joint": HAND2ARM_JOINT,
            "handbase": HAND_BASE,
            "grasp_tcp": GRASP_TCP,
        }

    gap = proxy.preinsert_gap if preinsert_gap_m is None else float(
        preinsert_gap_m
    )
    requested_inhand = post_grasp_error or PostGraspError()
    root = UsdGeom.Xform.Define(stage, V2_ROOT)
    root.GetPrim().GetReferences().AddReference(
        str(Path(v2_asset).resolve()), "/World/D38999InsertProxyV2"
    )
    tcp = np.asarray(tcp_transform, dtype=np.float64)
    initial_tcp = np.asarray(initial_tcp_transform, dtype=np.float64)
    plug_rotation = tcp[:3, :3]
    error_translation = np.asarray(
        receptacle_error_translation_assembly_m, dtype=np.float64
    )
    error_rotation_xyz = np.asarray(
        receptacle_error_rotation_xyz_rad, dtype=np.float64
    )
    if error_translation.shape != (3,) or error_rotation_xyz.shape != (3,):
        raise ValueError("receptacle error vectors must have length three")
    rotation = plug_rotation @ _rotation_xyz(*error_rotation_xyz)
    tcp_position = tcp[:3, 3]
    # Asset-local +Z is insertion.  The Plug Body origin is gap metres behind
    # the receptacle mouth.  The fixed latch locates that body origin a known
    # distance beyond grasp_tcp along the same local +Z axis.
    root_position = (
        tcp_position
        + plug_rotation
        @ np.asarray((0.0, 0.0, float(latch_tcp_to_body_m) + gap))
        + plug_rotation @ error_translation
    )
    quat = _quat_wxyz_from_rotation(rotation)
    xform = UsdGeom.Xformable(root)
    xform.AddTranslateOp().Set(Gf.Vec3d(*root_position))
    xform.AddOrientOp().Set(
        Gf.Quatf(float(quat[0]), Gf.Vec3f(*quat[1:]))
    )
    # The articulation's first reset is the canonical zero-arm state.  Author
    # PlugBody at that matching grasp-latch pose and then move the robot by
    # joint commands to the final preinsert pose.  This avoids both an initial
    # constraint impulse and any post-start object pose write.
    initial_world_from_tcp = np.asarray(initial_tcp, dtype=np.float64)
    initial_tcp_from_body = compose_nominal_with_error(
        (0.0, 0.0, float(latch_tcp_to_body_m)), requested_inhand
    )
    initial_body_world = initial_world_from_tcp @ initial_tcp_from_body
    root_world = np.eye(4, dtype=np.float64)
    root_world[:3, :3] = rotation
    root_world[:3, 3] = root_position
    root_from_body = np.linalg.inv(root_world) @ initial_body_world
    plug_quat = _quat_wxyz_from_rotation(root_from_body[:3, :3])
    # Override the asset's authored preinsert transform with the initial
    # zero-arm Plug pose.  Physics owns it after reset.
    plug_prim = stage.GetPrimAtPath(PLUG)
    translate = plug_prim.GetAttribute("xformOp:translate")
    if not translate.IsValid():
        translate = UsdGeom.Xformable(plug_prim).AddTranslateOp().GetAttr()
    translate.Set(Gf.Vec3d(*root_from_body[:3, 3]))
    plug_xform = UsdGeom.Xformable(plug_prim)
    plug_xform.AddOrientOp().Set(
        Gf.Quatf(float(plug_quat[0]), Gf.Vec3f(*plug_quat[1:]))
    )

    joints = UsdGeom.Xform.Define(stage, "/World/V2NominalJoints")
    joints.GetPrim().SetCustomDataByKey("kcg:mode", "SIM_DEBUG_GATE")
    latch = UsdPhysics.FixedJoint.Define(stage, GRASP_LATCH)
    latch.GetBody0Rel().SetTargets([Sdf.Path(HAND_BASE)])
    latch.GetBody1Rel().SetTargets([Sdf.Path(PLUG_BODY)])
    handbase_from_body = compose_nominal_with_error(
        (0.0, 0.0, float(0.4 + latch_tcp_to_body_m)), requested_inhand
    )
    latch_rotation = _quat_wxyz_from_rotation(handbase_from_body[:3, :3])
    latch.CreateLocalPos0Attr().Set(Gf.Vec3f(*handbase_from_body[:3, 3]))
    latch.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    latch.CreateLocalRot0Attr().Set(
        Gf.Quatf(
            float(latch_rotation[0]), Gf.Vec3f(*latch_rotation[1:])
        )
    )
    latch.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0)))
    latch.CreateCollisionEnabledAttr().Set(False)
    latch.GetPrim().SetCustomDataByKey("kcg:label", "GRASP_LATCH_PROXY")

    receptacle_prim = stage.GetPrimAtPath(RECEPTACLE)
    if not receptacle_prim.IsValid():
        raise RuntimeError("V2 receptacle is missing")
    UsdPhysics.RigidBodyAPI.Apply(receptacle_prim)
    mass = UsdPhysics.MassAPI.Apply(receptacle_prim)
    mass.CreateMassAttr().Set(1.0)
    fixed = UsdPhysics.FixedJoint.Define(stage, RECEPTACLE_WORLD_JOINT)
    # Empty body0 means world.  Its joint frame must be authored in world,
    # otherwise PhysX would pull the receptacle to the origin on reset.
    fixed.GetBody1Rel().SetTargets([Sdf.Path(RECEPTACLE)])
    fixed.CreateLocalPos0Attr().Set(Gf.Vec3f(*root_position))
    fixed.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    fixed.CreateLocalRot0Attr().Set(
        Gf.Quatf(float(quat[0]), Gf.Vec3f(*quat[1:]))
    )
    fixed.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0)))
    fixed.CreateCollisionEnabledAttr().Set(False)

    nut_joint = UsdPhysics.RevoluteJoint(
        stage.GetPrimAtPath(COUPLING_NUT_JOINT)
    )
    if not nut_joint or nut_joint.GetAxisAttr().Get() != "Z":
        raise RuntimeError("V2 CouplingNutJoint must be revolute about local Z")
    nut_joint.CreateCollisionEnabledAttr().Set(False)
    return {
        "mode": "REAL_IIWA_HAND_LOAD_PATH",
        "robot_root": ROBOT_ROOT,
        "articulation_root": ARTICULATION_ROOT,
        "hand2arm_joint": HAND2ARM_JOINT,
        "handbase": HAND_BASE,
        "grasp_tcp": GRASP_TCP,
        "grasp_latch_proxy": GRASP_LATCH,
        "plug_body": PLUG_BODY,
        "coupling_nut_joint": COUPLING_NUT_JOINT,
        "coupling_nut": COUPLING_NUT,
        "receptacle": RECEPTACLE,
        "receptacle_world_joint": RECEPTACLE_WORLD_JOINT,
        "root_position_world_m": [float(value) for value in root_position],
        "root_rotation_wxyz": [float(value) for value in quat],
        "initial_plug_root_translation_m": [
            float(value) for value in root_from_body[:3, 3]
        ],
        "initial_plug_root_rotation_wxyz": [
            float(value) for value in plug_quat
        ],
        "preinsert_gap_m": gap,
        "latch_tcp_to_body_m": float(latch_tcp_to_body_m),
        "requested_inhand_error": {
            "translation_m": [
                float(value) for value in requested_inhand.translation_m
            ],
            "rotation_xyz_rad": [
                float(value) for value in requested_inhand.rotation_xyz_rad
            ],
            "source": requested_inhand.source,
        },
        "authored_handbase_to_plug_body_translation_m": [
            float(value) for value in handbase_from_body[:3, 3]
        ],
        "authored_handbase_to_plug_body_rotation_wxyz": [
            float(value) for value in latch_rotation
        ],
        "grasp_latch_label": "GRASP_LATCH_PROXY",
        "receptacle_error_translation_assembly_m": [
            float(value) for value in error_translation
        ],
        "receptacle_error_rotation_xyz_rad": [
            float(value) for value in error_rotation_xyz
        ],
        "thread_proxy_enabled_during_insert": False,
        "rack_proxy_enabled_during_insert": False,
        "brake_proxy_enabled_during_insert": False,
        "anti_spin_proxy_enabled_during_insert": False,
        "standalone_plug_actuator": False,
        "standalone_nut_actuator": False,
    }


def apply_diagnostic_collision_profile(stage, UsdPhysics, profile: str):
    """Author a V2 collision-isolation profile before physics starts.

    These profiles exist only to identify the first blocking geometry.  The
    selected collider paths are returned for post-hoc reporting and are never
    exposed to the robot controller.
    """

    if profile not in COLLISION_PROFILES:
        raise ValueError(f"unknown V2 collision profile: {profile}")

    def group(path: str) -> str | None:
        if path.startswith(PLUG_BODY + "/NoseChamfer/"):
            return "plug_mating_nose"
        if path.startswith(PLUG_BODY + "/GuideShell/"):
            return "plug_guide_shell"
        if path.startswith(PLUG_BODY + "/C2Keys/"):
            return "plug_c2_keys"
        if path.startswith(PLUG_BODY + "/RearBody"):
            return "plug_outer_rear_body"
        if path.startswith(COUPLING_NUT + "/"):
            return "coupling_nut"
        if path.startswith(RECEPTACLE + "/EntranceChamfer/"):
            return "receptacle_mating_chamfer"
        if path.startswith(RECEPTACLE + "/GuideBore/"):
            return "receptacle_guide_bore"
        if path.startswith(RECEPTACLE + "/RearBody"):
            return "receptacle_outer_rear_body"
        if path.startswith(RECEPTACLE + "/Flange"):
            return "receptacle_flange"
        return None

    disabled_groups = {
        "full": set(),
        "nose_guide_only": {
            "plug_c2_keys",
            "plug_outer_rear_body",
            "coupling_nut",
            "receptacle_outer_rear_body",
            "receptacle_flange",
        },
        "no_coupling_nut": {"coupling_nut"},
        "no_c2_keys": {"plug_c2_keys"},
        "no_outer_nonmating": {
            "plug_outer_rear_body",
            "receptacle_outer_rear_body",
            "receptacle_flange",
        },
    }[profile]
    enabled_paths = []
    disabled_paths = []
    unclassified_paths = []
    groups = {}
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(V2_ROOT + "/") or not prim.HasAPI(
            UsdPhysics.CollisionAPI
        ):
            continue
        collider_group = group(path)
        if collider_group is None:
            unclassified_paths.append(path)
            continue
        enabled = collider_group not in disabled_groups
        collision = UsdPhysics.CollisionAPI(prim)
        collision.CreateCollisionEnabledAttr().Set(enabled)
        groups[collider_group] = groups.get(collider_group, 0) + 1
        (enabled_paths if enabled else disabled_paths).append(path)
    if unclassified_paths:
        raise RuntimeError(
            "unclassified V2 collision paths: " + repr(unclassified_paths)
        )
    if not enabled_paths:
        raise RuntimeError("collision isolation disabled every V2 collider")
    return {
        "label": "POSTHOC_DIAGNOSTIC",
        "profile": profile,
        "enabled_collider_paths": enabled_paths,
        "disabled_collider_paths": disabled_paths,
        "collider_group_counts": groups,
        "controller_input": False,
        "formal_pass_eligible": profile == "full",
    }


def topology_report(stage, UsdPhysics):
    def targets(joint, body):
        relation = joint.GetBody0Rel() if body == 0 else joint.GetBody1Rel()
        return [str(path) for path in relation.GetTargets()]

    hand2arm = UsdPhysics.FixedJoint(stage.GetPrimAtPath(HAND2ARM_JOINT))
    latch = UsdPhysics.FixedJoint(stage.GetPrimAtPath(GRASP_LATCH))
    nut = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(COUPLING_NUT_JOINT))
    receptacle = UsdPhysics.FixedJoint(
        stage.GetPrimAtPath(RECEPTACLE_WORLD_JOINT)
    )
    if not all((hand2arm, latch, nut, receptacle)):
        raise RuntimeError("required iiwa-hand-V2 joint topology is incomplete")
    world_to_plug = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.FixedJoint):
            continue
        joint = UsdPhysics.FixedJoint(prim)
        if targets(joint, 0) == [] and PLUG_BODY in targets(joint, 1):
            world_to_plug.append(str(prim.GetPath()))
    return {
        "hand2arm": {
            "joint_type": "FixedJoint",
            "body0": targets(hand2arm, 0),
            "body1": targets(hand2arm, 1),
        },
        "grasp_latch_proxy": {
            "joint_type": "FixedJoint",
            "label": latch.GetPrim().GetCustomDataByKey("kcg:label"),
            "body0": targets(latch, 0),
            "body1": targets(latch, 1),
            "collision_enabled": bool(latch.GetCollisionEnabledAttr().Get()),
        },
        "coupling_nut_joint": {
            "joint_type": "RevoluteJoint",
            "body0": targets(nut, 0),
            "body1": targets(nut, 1),
            "axis": str(nut.GetAxisAttr().Get()),
            "collision_enabled": bool(nut.GetCollisionEnabledAttr().Get()),
            "drive_api_present": bool(
                nut.GetPrim().HasAPI(UsdPhysics.DriveAPI, "angular")
            ),
        },
        "receptacle_world_fixed": {
            "joint_type": "FixedJoint",
            "body0": targets(receptacle, 0),
            "body1": targets(receptacle, 1),
        },
        "world_to_plug_fixed_joints": world_to_plug,
        "standalone_actuator_present": False,
    }


__all__ = [
    "ARM_NAMES",
    "ARTICULATION_ROOT",
    "COLLISION_PROFILES",
    "COUPLING_NUT",
    "COUPLING_NUT_JOINT",
    "GRASP_LATCH",
    "GRASP_TCP",
    "HAND2ARM_JOINT",
    "HAND_BASE",
    "PLUG_BODY",
    "RECEPTACLE",
    "ROBOT_ROOT",
    "V2_ROOT",
    "apply_diagnostic_collision_profile",
    "author_scene",
    "topology_report",
]
