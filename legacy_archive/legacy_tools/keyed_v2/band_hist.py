#!/usr/bin/env python3
"""2D histogram (plug-frame z vs r) of observed surfaces at shell-band pixels."""
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
    t_hp = np.linalg.inv(t_wh) @ plug
    truth6 = np.concatenate((t_hp[:3,3], Rotation.from_matrix(t_hp[:3,:3]).as_euler("xyz")))
    twp = t_wh @ pose_matrix(truth6)
    shell = shell25j_plug_cad_profile(feature_set="shell_only")
    xyz = shell.plug_mating.xyz[shell.plug_mating.label == 1]
    world = twp[:3,:3] @ xyz.T + twp[:3,3].reshape(3,1)
    uv, pred = project(camera, world.T)
    u = np.rint(uv[:,0]).astype(np.int64); v = np.rint(uv[:,1]).astype(np.int64)
    ok = (pred > 0.03) & (u >= 0) & (u < 1280) & (v >= 0) & (v < 720)
    obs = depth[v[ok], u[ok]]
    # back-project observed via pixel rays
    fx, fy, cx, cy = 1466.0, 1466.0, 640.0, 360.0
    eye = np.asarray(cam["eye_m"])
    rows = np.asarray(camera.world_to_camera)
    right = rows[0]; up = -rows[1]; fwd = rows[2]
    xs = (uv[ok,0]-cx)/fx * obs
    ys = (uv[ok,1]-cy)/fy * obs
    pts = eye + right*xs[:,None] + up*ys[:,None] + fwd*obs[:,None]
    p_plug = (pts - plug[:3,3]) @ plug[:3,:3]
    zz = p_plug[:,2]*1000.0
    rr = np.linalg.norm(p_plug[:,:2], axis=1)*1000.0
    print("z histogram (mm) of observed surfaces at band pixels:")
    hist, edges = np.histogram(zz, bins=[-80,-40,-20,-10,-5,0,2,5,8,11,15,25,40])
    for h, lo, hi in zip(hist, edges[:-1], edges[1:]):
        print(f"  z[{lo:5.0f},{hi:5.0f}): {h:5d}  {100*h/len(zz):.1f}%")
    print(f"  r: med={np.median(rr):.1f} p10={np.percentile(rr,10):.1f} p90={np.percentile(rr,90):.1f}")
    # where are the points with z in [0,12] (the band zone)?
    m = (zz >= 0) & (zz <= 12)
    print(f"band-zone points ({np.sum(m)}): r med={np.median(rr[m]):.1f}  z med={np.median(zz[m]):.1f}")
    m2 = (zz < 0)
    print(f"below-face points ({np.sum(m2)}): r med={np.median(rr[m2]):.1f} z med={np.median(zz[m2]):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
