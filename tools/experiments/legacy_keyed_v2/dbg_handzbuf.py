#!/usr/bin/env python3
"""Debug: where does the hand z-buffer land relative to the face pixels?"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/home/noob/WorkPlace/kcgtest1/src/kcg_connector")
from kcg_connector.d38999_cad_registration import (
    CameraModel, fixed_camera_model, project, proxy_cad_points,
)
from kcg_connector.d38999_inhand_multiview import pose_matrix
from kcg_connector.hand_occluder_cad import build_hand_occluder_cad

BASE = Path("/home/noob/WorkPlace/kcgtest1/artifacts/kcg_connector/"
            "d38999_postgrasp_visual_ft_e2e_v1/phase8_visual_chain_v6")
SNAP = Path("/home/noob/WorkPlace/kcgtest1/artifacts/kcg_connector/"
            "d38999_postgrasp_visual_ft_e2e_v1/phase1_snapshot_gate_v3/seed000/"
            "postgrasp_snapshot_gate/snapshot_gate.json")
URDF = Path("/home/noob/WorkPlace/kcgtest1/artifacts/kcg_connector/urdf/handarm.urdf")
MESH = Path("/home/noob/WorkPlace/kcgtest1/src/iiwa_description/meshes/hand")


def main() -> int:
    snap = json.loads(SNAP.read_text())
    hand_q = snap["robot_state"]["q_rad"][7:15]
    hand_cad = build_hand_occluder_cad(hand_q, URDF, MESH)
    vd = BASE / "formal_views/PALM_H0_K0"
    cam = json.loads((vd / "camera.json").read_text())
    camera = fixed_camera_model(eye=tuple(cam["eye_m"]), target=tuple(cam["target_m"]),
                                resolution=(1280, 720))
    fk = json.loads((vd / "fk.json").read_text())
    t_wh = np.asarray(fk["T_WH_4x4"])
    # hand points in hand frame -> world -> camera
    world = hand_cad.xyz @ t_wh[:3,:3].T + t_wh[:3,3]
    uv, depth = project(camera, world)
    u = np.rint(uv[:,0]).astype(np.int64); v = np.rint(uv[:,1]).astype(np.int64)
    ok = (depth > 0.03) & (u >= 0) & (u < 1280) & (v >= 0) & (v < 720)
    print(f"hand points projected in-frame: {np.sum(ok)}/{len(hand_cad.xyz)}")
    print(f"hand depth range: {depth[ok].min():.3f}..{depth[ok].max():.3f}")
    print(f"hand pixel bbox: u[{u[ok].min()},{u[ok].max()}] v[{v[ok].min()},{v[ok].max()}]")
    # face points at truth
    side = json.loads((BASE / "posthoc_truth_sidecar.json").read_text())
    qs = np.asarray(side["plug_orientation_wxyz"])
    rot = Rotation.from_quat([qs[1], qs[2], qs[3], qs[0]])
    plug = np.eye(4); plug[:3,:3] = rot.as_matrix()
    plug[:3,3] = np.asarray(side["plug_position_m"])
    t_hp = np.linalg.inv(t_wh) @ plug
    truth6 = np.concatenate((t_hp[:3,3], Rotation.from_matrix(t_hp[:3,:3]).as_euler("xyz")))
    plug_cad, _ = proxy_cad_points()
    xyz = plug_cad.xyz[plug_cad.label == 1]
    twp = t_wh @ pose_matrix(truth6)
    w = twp[:3,:3] @ xyz.T + twp[:3,3].reshape(3,1)
    uvf, predf = project(camera, w.T)
    uf = np.rint(uvf[:,0]).astype(np.int64); vf = np.rint(uvf[:,1]).astype(np.int64)
    okf = (predf > 0.03) & (uf >= 0) & (uf < 1280) & (vf >= 0) & (vf < 720)
    # hand z-buffer
    img = np.full((720, 1280), np.inf)
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(-depth[idx])]
    img[v[order], u[order]] = depth[order]
    # face pixels where hand is in front
    hand_front = img[vf[okf], uf[okf]] < predf[okf] - 0.0005
    print(f"face points: {np.sum(okf)}, hand-in-front: {np.sum(hand_front)}")
    print(f"face depth range: {predf[okf].min():.3f}..{predf[okf].max():.3f}")
    # z-buffer coverage: where is img finite in the face bbox?
    bbox_u = uf[okf].min(), uf[okf].max()
    bbox_v = vf[okf].min(), vf[okf].max()
    sub = img[bbox_v[0]:bbox_v[1]+1, bbox_u[0]:bbox_u[1]+1]
    print(f"face bbox: u{bbox_u} v{bbox_v}; hand z-buffer coverage in bbox: {np.isfinite(sub).mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
