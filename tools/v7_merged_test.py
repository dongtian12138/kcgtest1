#!/usr/bin/env python3
"""Merged palm profile: legacy rings (tilt anchor) + current-asset band walls (z anchor)."""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/home/noob/WorkPlace/kcgtest1/src/kcg_connector")
from kcg_connector.d38999_cad_registration import (
    PLUG_MATING, CameraModel, CadPoints, fixed_camera_model,
    shell25j_plug_cad_profile,
)
from kcg_connector.d38999_inhand_multiview import pose_matrix
from kcg_connector.hand_occluder_cad import build_hand_occluder_cad
from kcg_connector.postgrasp_shadow_estimator import (
    FormalView, estimate_postgrasp_T_HP,
)

BASE = Path("/home/noob/WorkPlace/kcgtest1/artifacts/kcg_connector/"
            "d38999_postgrasp_visual_ft_e2e_v1/phase8_visual_chain_v7")
SNAP = Path("/home/noob/WorkPlace/kcgtest1/artifacts/kcg_connector/"
            "d38999_postgrasp_visual_ft_e2e_v1/phase1_snapshot_gate_v3/seed000/"
            "postgrasp_snapshot_gate/snapshot_gate.json")
URDF = Path("/home/noob/WorkPlace/kcgtest1/artifacts/kcg_connector/urdf/handarm.urdf")
MESH = Path("/home/noob/WorkPlace/kcgtest1/src/iiwa_description/meshes/hand")


def _ring(radius, z, count=240, edge=True):
    angle = np.linspace(0.0, 2.0*math.pi, count, endpoint=False)
    xyz = np.column_stack((radius*np.cos(angle), radius*np.sin(angle), np.full(count, z)))
    normal = np.tile((0.0, 0.0, -1.0), (count, 1))
    return xyz, normal, np.full(count, PLUG_MATING), np.full(count, edge)


def _cylinder(radius, z0, z1, azimuth=240, layers=6):
    angle = np.linspace(0.0, 2.0*math.pi, azimuth, endpoint=False)
    z = np.linspace(z0, z1, layers)
    aa, zz = np.meshgrid(angle, z)
    xyz = np.column_stack((radius*np.cos(aa.ravel()), radius*np.sin(aa.ravel()), zz.ravel()))
    normal = np.column_stack((np.cos(aa.ravel()), np.sin(aa.ravel()), np.zeros(aa.size)))
    edge = np.isclose(zz.ravel(), z0) | np.isclose(zz.ravel(), z1)
    return xyz, normal, np.full(len(xyz), PLUG_MATING), edge


def merged_palm_cad() -> CadPoints:
    parts = [
        _ring(0.0182, 0.0, edge=True),   # face outer rim (tilt anchor)
        _ring(0.0165, 0.0, edge=True),   # face inner rim (tilt anchor)
        _cylinder(0.0165, 0.0005, 0.0095),  # band inner wall (z anchor)
        _cylinder(0.0190, 0.0005, 0.0095),  # band outer wall (z anchor)
        _ring(0.01775, 0.010, count=360, edge=True),  # band top rim
    ]
    xyz = np.concatenate([p[0] for p in parts], axis=0)
    normal = np.concatenate([p[1] for p in parts], axis=0)
    label = np.concatenate([p[2] for p in parts], axis=0)
    edge = np.concatenate([p[3] for p in parts], axis=0)
    return CadPoints(xyz, normal, label.astype(np.int16), edge.astype(bool))


def load_view(view_id, group):
    vd = BASE / "formal_views" / view_id
    rgb = cv2.cvtColor(cv2.imread(str(vd / "rgb.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    depth = np.load(vd / "depth_m.npy").astype(np.float32)
    cam = json.loads((vd / "camera.json").read_text())
    model = fixed_camera_model(eye=tuple(cam["eye_m"]), target=tuple(cam["target_m"]),
                               resolution=(1280, 720))
    intr = np.asarray(cam["intrinsics"])
    camera = CameraModel(1280, 720, float(intr[0,0]), float(intr[1,1]),
                         float(intr[0,2]), float(intr[1,2]),
                         tuple(model.position_world), tuple(model.world_to_camera))
    fk = json.loads((vd / "fk.json").read_text())
    return FormalView(view_id=view_id, timestamp_utc="t", rgb=rgb, depth=depth,
                      camera=camera, T_WH=np.asarray(fk["T_WH_4x4"]),
                      T_WC=np.asarray(fk["T_WC_4x4"]), group=group,
                      extrinsic_source="T_HC_calibrated")


def main() -> int:
    snap = json.loads(SNAP.read_text())
    hand_cad = build_hand_occluder_cad(snap["robot_state"]["q_rad"][7:15], URDF, MESH)
    side = json.loads((BASE / "posthoc_truth_sidecar.json").read_text())
    qs = np.asarray(side["plug_orientation_wxyz"])
    rot = Rotation.from_quat([qs[1], qs[2], qs[3], qs[0]])
    plug = np.eye(4); plug[:3,:3] = rot.as_matrix()
    plug[:3,3] = np.asarray(side["plug_position_m"])
    palm_fk = json.loads((BASE / "formal_views/PALM_H0_K0/fk.json").read_text())
    t_wh = np.asarray(palm_fk["T_WH_4x4"])
    t_hp = np.linalg.inv(t_wh) @ plug
    truth6 = np.concatenate((t_hp[:3,3], Rotation.from_matrix(t_hp[:3,:3]).as_euler("xyz")))
    print(f"truth xyz(mm): {(truth6[:3]*1000).round(2)}")
    initial = np.zeros(12)
    initial[:6] = np.array([0.0, 0.0, 0.4485, math.pi, 0.0, 0.0])
    palm = load_view("PALM_H0_K0", "postgrasp_inhand_views")
    wrist = load_view("WRIST_H0", "postgrasp_second_inhand_camera_views")
    shell = shell25j_plug_cad_profile(feature_set="shell_plus_socket")
    merged = merged_palm_cad()
    for label, views in (("merged_joint", [palm, wrist]), ("merged_palm", [palm])):
        result = estimate_postgrasp_T_HP(
            views, initial, plug_cad=merged, receptacle_cad=shell.receptacle,
            plug_occluder_cad=shell.plug_occluders, hand_occluder_cad=hand_cad,
            occlusion_policy="baseline", edge_policy="depth_gated",
            optimizer_variant="multistart_physical_jacobian", multistart_count=17)
        print(f"=== {label} ===")
        for hyp in result["c2"]["hypotheses"]:
            est = np.asarray(hyp["T_hand_plug_xyz_rpy"])
            err = np.linalg.norm(est[:3]-truth6[:3])*1000
            dz = (est[2]-truth6[2])*1000
            axis_e = pose_matrix(est)[:3,:3] @ np.array([0,0,1.0]); axis_e/=np.linalg.norm(axis_e)
            tilt = np.degrees(np.arccos(abs(axis_e[2])))
            print(f"  {hyp['id']}: err={err:.2f}mm dz={dz:+.2f}mm tilt={tilt:.2f}deg rms={hyp['residual_rms']:.2f} "
                  f"support_fail={hyp['plug_support_gate_failed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
