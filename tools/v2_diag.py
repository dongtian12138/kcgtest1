#!/usr/bin/env python3
"""Decisive depth diagnostic on the fresh v2 palm capture at capture-time truth."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/home/noob/WorkPlace/kcgtest1/src/kcg_connector")
from kcg_connector.d38999_cad_registration import (
    fixed_camera_model, project, shell25j_plug_cad_profile,
)
from kcg_connector.d38999_inhand_multiview import pose_matrix

BASE = Path("/home/noob/WorkPlace/kcgtest1/artifacts/kcg_connector/"
            "d38999_postgrasp_visual_ft_e2e_v1")
ROOT = BASE / "phase7_palm_wrist_joint_v2/formal_views/PALM_H0_K0"
SIDE = BASE / "phase7_palm_wrist_joint_v2/posthoc_truth_sidecar.json"


def main() -> int:
    depth = np.load(ROOT / "depth_m.npy").astype(np.float64)
    cam = json.loads((ROOT / "camera.json").read_text())
    camera = fixed_camera_model(eye=tuple(cam["eye_m"]), target=tuple(cam["target_m"]),
                                resolution=(1280, 720))
    fk = json.loads((ROOT / "fk.json").read_text())
    t_wh = np.asarray(fk["T_WH_4x4"])
    side = json.loads(SIDE.read_text())
    qs = np.asarray(side["plug_orientation_wxyz"])
    rot = Rotation.from_quat([qs[1], qs[2], qs[3], qs[0]])
    plug = np.eye(4); plug[:3,:3] = rot.as_matrix()
    plug[:3,3] = np.asarray(side["plug_position_m"])
    face_normal_world = plug[:3,:3] @ np.array([0,0,-1.0])
    print("face outward normal (world):", face_normal_world.round(4), "(z>0 = face UP)")
    t_hp = np.linalg.inv(t_wh) @ plug
    truth6 = np.concatenate((t_hp[:3,3], Rotation.from_matrix(t_hp[:3,:3]).as_euler("xyz")))
    twp = t_wh @ pose_matrix(truth6)
    shell = shell25j_plug_cad_profile(feature_set="shell_plus_socket")
    cad = shell.plug_mating
    sel = cad.label == 1
    xyz = cad.xyz[sel]
    world = twp[:3,:3] @ xyz.T + twp[:3,3].reshape(3,1)
    uv, pred = project(camera, world.T)
    u = np.rint(uv[:,0]).astype(np.int64); v = np.rint(uv[:,1]).astype(np.int64)
    ok = (pred > 0.03) & (u >= 0) & (u < 1280) & (v >= 0) & (v < 720)
    obs = depth[v[ok], u[ok]]
    dvalid = np.isfinite(obs) & (obs > 0.0)
    diff = (pred[ok][dvalid] - obs[dvalid]) * 1000.0
    print(f"n={np.sum(ok)} depth_valid={np.sum(dvalid)}")
    print(f"pred depth: med={np.median(pred[ok][dvalid]):.3f}  obs depth: med={np.median(obs[dvalid]):.3f}")
    print(f"diff(pred-obs) mm: p10={np.percentile(diff,10):.1f} med={np.median(diff):.1f} p90={np.percentile(diff,90):.1f}")
    print(f"|diff|<2mm: {np.mean(np.abs(diff)<2)*100:.1f}%  <5mm: {np.mean(np.abs(diff)<5)*100:.1f}%  >20mm: {np.mean(np.abs(diff)>20)*100:.1f}%")
    # split by plug-frame radius of the CAD source points
    r_plug = np.linalg.norm(xyz[ok][dvalid][:, :2], axis=1) * 1000.0
    for lo, hi in ((0, 15), (15, 17), (17, 20)):
        m = (r_plug >= lo) & (r_plug < hi)
        if np.any(m):
            print(f"  r[{lo:2d},{hi:2d})mm: n={np.sum(m)} med_diff={np.median(diff[m]):.1f}mm |d|<5: {np.mean(np.abs(diff[m])<5)*100:.0f}%")
    # back-project the visible subset (|diff|<5mm) -> where are they
    vis = np.abs(diff) < 5
    fx, fy, cx, cy = 1466.0, 1466.0, 640.0, 360.0
    right = np.asarray(camera.world_to_camera)[0]; up = -np.asarray(camera.world_to_camera)[1]
    fwd = np.asarray(camera.world_to_camera)[2]
    eye = np.asarray(cam["eye_m"])
    xs = (uv[ok][dvalid][vis,0]-cx)/fx * obs[dvalid][vis]
    ys = (uv[ok][dvalid][vis,1]-cy)/fy * obs[dvalid][vis]
    pts = eye + right*xs[:,None] + up*ys[:,None] + fwd*obs[dvalid][vis,None]
    p_plug = (pts - plug[:3,3]) @ plug[:3,:3]
    zz = p_plug[:,2]*1000; rr = np.linalg.norm(p_plug[:,:2],axis=1)*1000
    print(f"visible pts (n={len(zz)}): plug-frame z med={np.median(zz):.1f}mm r med={np.median(rr):.1f}mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
