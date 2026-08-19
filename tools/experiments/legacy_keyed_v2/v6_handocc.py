#!/usr/bin/env python3
"""Test hand-occluder support on v6 data (CPU)."""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/home/noob/WorkPlace/kcgtest1/src/kcg_connector")
from kcg_connector.d38999_cad_registration import (
    CameraModel, fixed_camera_model, proxy_cad_points,
    shell25j_plug_cad_profile,
)
from kcg_connector.d38999_inhand_multiview import pose_matrix
from kcg_connector.hand_occluder_cad import build_hand_occluder_cad
from kcg_connector.postgrasp_shadow_estimator import (
    FormalView, estimate_postgrasp_T_HP,
)

BASE = Path("/home/noob/WorkPlace/kcgtest1/artifacts/kcg_connector/"
            "d38999_postgrasp_visual_ft_e2e_v1/phase8_visual_chain_v6")
SNAP = Path("/home/noob/WorkPlace/kcgtest1/artifacts/kcg_connector/"
            "d38999_postgrasp_visual_ft_e2e_v1/phase1_snapshot_gate_v3/seed000/"
            "postgrasp_snapshot_gate/snapshot_gate.json")
URDF = Path("/home/noob/WorkPlace/kcgtest1/artifacts/kcg_connector/urdf/handarm.urdf")
MESH = Path("/home/noob/WorkPlace/kcgtest1/src/iiwa_description/meshes/hand")


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
    hand_q = snap["robot_state"]["q_rad"][7:15]
    hand_cad = build_hand_occluder_cad(hand_q, URDF, MESH)
    print(f"hand occluder points: {len(hand_cad.xyz)}")
    side = json.loads((BASE / "posthoc_truth_sidecar.json").read_text())
    qs = np.asarray(side["plug_orientation_wxyz"])
    rot = Rotation.from_quat([qs[1], qs[2], qs[3], qs[0]])
    plug = np.eye(4); plug[:3,:3] = rot.as_matrix()
    plug[:3,3] = np.asarray(side["plug_position_m"])
    palm_fk = json.loads((BASE / "formal_views/PALM_H0_K0/fk.json").read_text())
    t_wh = np.asarray(palm_fk["T_WH_4x4"])
    t_hp = np.linalg.inv(t_wh) @ plug
    truth6 = np.concatenate((t_hp[:3,3], Rotation.from_matrix(t_hp[:3,:3]).as_euler("xyz")))
    axis_t = t_hp[:3,:3] @ np.array([0,0,1.0]); axis_t/=np.linalg.norm(axis_t)
    print(f"truth: xyz(mm)={(truth6[:3]*1000).round(2)} tilt={np.degrees(np.arccos(abs(axis_t[2]))):.2f}deg")
    initial = np.zeros(12)
    initial[:6] = np.array([0.0, 0.0, 0.4485, math.pi, 0.0, 0.0])
    palm = load_view("PALM_H0_K0", "postgrasp_inhand_views")
    wrist = load_view("WRIST_H0", "postgrasp_second_inhand_camera_views")
    legacy_plug, legacy_rec = proxy_cad_points()
    shell = shell25j_plug_cad_profile(feature_set="shell_plus_socket")
    for label, views, policy in (
        ("joint_handocc_baseline", [palm, wrist], "baseline"),
        ("joint_handocc_ignore", [palm, wrist], "ignore_foreground_occluded"),
        ("palm_handocc_baseline", [palm], "baseline"),
    ):
        result = estimate_postgrasp_T_HP(
            views, initial, plug_cad=legacy_plug, receptacle_cad=legacy_rec,
            plug_occluder_cad=shell.plug_occluders,
            hand_occluder_cad=hand_cad,
            occlusion_policy=policy, edge_policy="depth_gated",
            optimizer_variant="multistart_physical_jacobian", multistart_count=17)
        print(f"=== {label} ===")
        for hyp in result["c2"]["hypotheses"]:
            est = np.asarray(hyp["T_hand_plug_xyz_rpy"])
            err = np.linalg.norm(est[:3]-truth6[:3])*1000
            axis_e = pose_matrix(est)[:3,:3] @ np.array([0,0,1.0]); axis_e/=np.linalg.norm(axis_e)
            tilt = np.degrees(np.arccos(abs(axis_e[2])))
            print(f"  {hyp['id']}: err={err:.2f}mm tilt={tilt:.2f}deg rms={hyp['residual_rms']:.2f} "
                  f"support_fail={hyp['plug_support_gate_failed']}")
            for d in hyp.get("plug_support_diagnostics") or []:
                print(f"    {d['view_id']}: vis={d['visible_depth_support_fraction']:.3f} "
                      f"adj={d.get('visible_depth_support_fraction_occluder_adjusted', float('nan')):.3f} "
                      f"fg_occ={d['foreground_occluded_fraction']:.3f} "
                      f"cad_occ={d['cad_occluder_fraction']:.3f} missing={d['missing_surface_fraction']:.3f}")
        print(f"  pose_valid={result['pose_valid']} reasons={result['pose_valid_reasons']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
