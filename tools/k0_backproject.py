import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/home/noob/WorkPlace/kcgtest1/src/kcg_connector")
from kcg_connector.d38999_cad_registration import fixed_camera_model, project, shell25j_plug_cad_profile
from kcg_connector.d38999_inhand_multiview import pose_matrix

BASE = Path("/home/noob/WorkPlace/kcgtest1/artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1")
SNAP = BASE / "phase1_snapshot_gate_v3/seed000/postgrasp_snapshot_gate/snapshot_gate.json"


def main() -> int:
    snap = json.loads(SNAP.read_text())
    q = np.asarray(snap["plug_root_state"]["orientation_wxyz"])
    rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])
    plug = np.eye(4); plug[:3,:3] = rot.as_matrix()
    plug[:3,3] = np.asarray(snap["plug_root_state"]["position_m"])
    t_pw = np.linalg.inv(plug)
    shell = shell25j_plug_cad_profile(feature_set="shell_only")
    cad = shell.plug_mating
    xyz = cad.xyz[cad.label == 1]
    for vid in ("PALM_H0_K0", "PALM_H0_K1", "PALM_H0_K2", "PALM_H0_K3", "PALM_H0_K4"):
        root = BASE / "phase7_palm_batch_v1/formal_views" / vid
        cam = json.loads((root / "camera.json").read_text())
        fk = json.loads((root / "fk.json").read_text())
        depth = np.load(root / "depth_m.npy").astype(np.float64)
        camera = fixed_camera_model(eye=tuple(cam["eye_m"]), target=tuple(cam["target_m"]), resolution=(1280, 720))
        t_wh = np.asarray(fk["T_WH_4x4"])
        t_hp = np.linalg.inv(t_wh) @ plug
        truth6 = np.concatenate((t_hp[:3,3], Rotation.from_matrix(t_hp[:3,:3]).as_euler("xyz")))
        twp = t_wh @ pose_matrix(truth6)
        world = twp[:3,:3] @ xyz.T + twp[:3,3].reshape(3,1)
        uv, pred = project(camera, world.T)
        u = np.rint(uv[:,0]).astype(np.int64); v = np.rint(uv[:,1]).astype(np.int64)
        ok = (pred > 0.03) & (u >= 0) & (u < 1280) & (v >= 0) & (v < 720)
        obs = depth[v[ok], u[ok]]
        # strict back-projection per pixel ray
        fx, fy, cx, cy = 1466.0, 1466.0, 640.0, 360.0
        eye = np.asarray(cam["eye_m"])
        rot_w2c = np.asarray(camera.world_to_camera)
        # camera x,y in world: rows of world_to_camera are (right, -up, forward)
        right = rot_w2c[0]; down = -rot_w2c[1]; forward = rot_w2c[2]
        xs = (uv[ok,0] - cx) / fx * obs
        ys = (uv[ok,1] - cy) / fy * obs
        pts_alt = {}
        for sx in (1.0, -1.0):
            for sy in (1.0, -1.0):
                pts_alt[(sx, sy)] = eye + sx*right * xs[:,None] + sy*down * ys[:,None] + forward * obs[:,None]
        pts = pts_alt[(1.0, 1.0)]
        for (sx, sy), p in pts_alt.items():
            p_plug = (p - plug[:3,3]) @ plug[:3,:3]
            zz = p_plug[:,2] * 1000.0
            rr = np.linalg.norm(p_plug[:,:2], axis=1) * 1000.0
            frac = np.mean((zz>=0)&(zz<60)&(rr>10)&(rr<40))
            print(f"=== {vid} sign=({sx:+.0f},{sy:+.0f}): z_med={np.median(zz):.1f}mm r_med={np.median(rr):.1f}mm near_face_frac={frac:.2f}")
        pts_plug = None

    return 0

if __name__ == "__main__":
    sys.exit(main())
