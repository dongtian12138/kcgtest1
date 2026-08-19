#!/usr/bin/env python3
"""Analyze phase7_palm_batch_v2 views with legacy+ignore+missing config."""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/home/noob/WorkPlace/kcgtest1/src/kcg_connector")
from kcg_connector.d38999_cad_registration import CameraModel, fixed_camera_model
from kcg_connector.d38999_inhand_multiview import pose_matrix
from kcg_connector.postgrasp_shadow_estimator import (
    FormalView, estimate_postgrasp_T_HP,
)

BASE = Path("/home/noob/WorkPlace/kcgtest1/artifacts/kcg_connector/"
            "d38999_postgrasp_visual_ft_e2e_v1/phase7_palm_batch_v2")


def load_view(view_id):
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
                      T_WC=np.asarray(fk["T_WC_4x4"]), group="postgrasp_inhand_views",
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
    axis_t = t_hp[:3,:3] @ np.array([0,0,1.0]); axis_t/=np.linalg.norm(axis_t)
    print(f"truth: xyz(mm)={(truth6[:3]*1000).round(2)} tilt={np.degrees(np.arccos(abs(axis_t[2]))):.2f}deg")
    initial = np.zeros(12)
    initial[:6] = np.array([0.0, 0.0, 0.4485, math.pi, 0.0, 0.0])
    view_ids = sorted(p.name for p in (BASE / "formal_views").iterdir() if p.is_dir())
    for vid in view_ids:
        view = load_view(vid)
        result = estimate_postgrasp_T_HP(
            [view], initial, cad_profile="legacy_axisymmetric",
            occlusion_policy="baseline", edge_policy="depth_gated",
            optimizer_variant="multistart_physical_jacobian", multistart_count=17)
        hyp = result["c2"]["hypotheses"][0]
        est = np.asarray(hyp["T_hand_plug_xyz_rpy"])
        err = np.linalg.norm(est[:3]-truth6[:3])*1000
        axis_e = pose_matrix(est)[:3,:3] @ np.array([0,0,1.0]); axis_e/=np.linalg.norm(axis_e)
        tilt = np.degrees(np.arccos(abs(axis_e[2])))
        sup = (hyp.get("plug_support_diagnostics") or [{}])[0]
        print(f"{vid}: err={err:.2f}mm tilt={tilt:.2f}deg rms={hyp['residual_rms']:.2f} "
              f"vis={sup.get('visible_depth_support_fraction', float('nan')):.3f} "
              f"fg_occ={sup.get('foreground_occluded_fraction', float('nan')):.3f} "
              f"missing={sup.get('missing_surface_fraction', float('nan')):.3f} "
              f"edge={sup.get('edge_support_fraction', float('nan')):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
