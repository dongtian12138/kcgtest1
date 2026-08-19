#!/usr/bin/env python3
"""Evaluate residual at the truth pose vs the solved pose for batch v2 views."""
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
)
from kcg_connector.postgrasp_shadow_estimator import _ResidualProblem

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
    return camera, np.asarray(fk["T_WH_4x4"])


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
    plug_cad, receptacle_cad = proxy_cad_points()
    from kcg_connector.postgrasp_shadow_estimator import FormalView
    for vid in ("PALM_H0_K0", "PALM_H0_K1", "PALM_H0_K2", "PALM_H0_K3", "PALM_H0_K4"):
        vd = BASE / "formal_views" / vid
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
        view = FormalView(view_id=vid, timestamp_utc="t", rgb=rgb, depth=depth,
                          camera=camera, T_WH=np.asarray(fk["T_WH_4x4"]),
                          T_WC=np.asarray(fk["T_WC_4x4"]), group="postgrasp_inhand_views",
                          extrinsic_source="T_HC_calibrated")
        state = np.zeros(12); state[:6] = truth6
        for policy in ("baseline", "ignore_foreground_occluded"):
            prob = _ResidualProblem([view], plug_cad, receptacle_cad, state,
                                    frozen_mask=(False,)*6+(True,)*6,
                                    endpoints=("plug",), occlusion_policy=policy,
                                    edge_policy="depth_gated", include_prior=False,
                                    missing_surface_margin_m=0.015)
            res = prob.residual(state)
            rms = float(np.sqrt(np.mean(res**2)))
            d = prob.last_plug_support[0]
            print(f"{vid} {policy}: truth_rms={rms:.3f} vis={d['visible_depth_support_fraction']:.3f} "
                  f"fg_occ={d['foreground_occluded_fraction']:.3f} missing={d['missing_surface_fraction']:.3f} "
                  f"edge={d['edge_support_fraction']:.3f} dgate={d['depth_gated_edge_support_fraction']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
