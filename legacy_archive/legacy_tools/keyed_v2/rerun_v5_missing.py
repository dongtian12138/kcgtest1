#!/usr/bin/env python3
"""Re-run joint T_HP on the v5 chain captures with missing-surface handling."""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/home/noob/WorkPlace/kcgtest1/src/kcg_connector")
from kcg_connector.d38999_cad_registration import CameraModel, fixed_camera_model
from kcg_connector.postgrasp_shadow_estimator import (
    FormalView, estimate_postgrasp_T_HP,
)

BASE = Path("/home/noob/WorkPlace/kcgtest1/artifacts/kcg_connector/"
            "d38999_postgrasp_visual_ft_e2e_v1/phase8_visual_chain_v5")


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
    side = json.loads((BASE / "posthoc_truth_sidecar.json").read_text())
    qs = np.asarray(side["plug_orientation_wxyz"])
    rot = Rotation.from_quat([qs[1], qs[2], qs[3], qs[0]])
    plug = np.eye(4); plug[:3,:3] = rot.as_matrix()
    plug[:3,3] = np.asarray(side["plug_position_m"])
    palm_fk = json.loads((BASE / "formal_views/PALM_H0_K0/fk.json").read_text())
    t_wh = np.asarray(palm_fk["T_WH_4x4"])
    t_hp = np.linalg.inv(t_wh) @ plug
    truth6 = np.concatenate((t_hp[:3,3], Rotation.from_matrix(t_hp[:3,:3]).as_euler("xyz")))
    initial = np.zeros(12)
    initial[:6] = np.array([0.0, 0.0, 0.4485, math.pi, 0.0, 0.0])
    views = [load_view("PALM_H0_K0", "postgrasp_inhand_views"),
             load_view("WRIST_H0", "postgrasp_second_inhand_camera_views")]
    for policy in ("baseline", "ignore_foreground_occluded"):
        result = estimate_postgrasp_T_HP(
            views, initial, cad_profile="legacy_axisymmetric",
            occlusion_policy=policy, edge_policy="depth_gated",
            optimizer_variant="multistart_physical_jacobian", multistart_count=17)
        print(f"=== occlusion_policy={policy} ===")
        for hyp in result["c2"]["hypotheses"]:
            est = np.asarray(hyp["T_hand_plug_xyz_rpy"])
            err = np.linalg.norm(est[:3]-truth6[:3])*1000
            print(f"  {hyp['id']}: err_mm={err:.2f} rms={hyp['residual_rms']:.3f} "
                  f"support_fail={hyp['plug_support_gate_failed']} cond={hyp['condition_number'] and round(hyp['condition_number'],1)}")
            for d in hyp.get("plug_support_diagnostics") or []:
                print(f"    {d['view_id']}: in_frame={d['in_frame_fraction']:.2f} vis={d['visible_depth_support_fraction']:.2f} "
                      f"fg_occ={d['foreground_occluded_fraction']:.2f} missing={d.get('missing_surface_fraction', float('nan')):.2f} "
                      f"edge={d['edge_support_fraction']:.2f} dgate={d['depth_gated_edge_support_fraction']:.2f}")
    for hyp in result["c2"]["hypotheses"]:
        est = np.asarray(hyp["T_hand_plug_xyz_rpy"])
        err = np.linalg.norm(est[:3]-truth6[:3])*1000
        print(f"{hyp['id']}: err_mm={err:.2f} rms={hyp['residual_rms']:.3f} "
              f"support_fail={hyp['plug_support_gate_failed']} cond={hyp['condition_number'] and round(hyp['condition_number'],1)}")
        for d in hyp.get("plug_support_diagnostics") or []:
            print(f"    {d['view_id']}: in_frame={d['in_frame_fraction']:.2f} vis={d['visible_depth_support_fraction']:.2f} "
                  f"fg_occ={d['foreground_occluded_fraction']:.2f} missing={d.get('missing_surface_fraction', float('nan')):.2f} "
                  f"edge={d['edge_support_fraction']:.2f} dgate={d['depth_gated_edge_support_fraction']:.2f}")
    print("truth xyz(mm):", (truth6[:3]*1000).round(2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
