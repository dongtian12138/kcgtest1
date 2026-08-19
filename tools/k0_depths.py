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
    for vid in ("PALM_H0_K0", "PALM_H0_K1", "PALM_H0_K2"):
        root = BASE / "phase7_palm_batch_v1/formal_views" / vid
        cam = json.loads((root / "camera.json").read_text())
        fk = json.loads((root / "fk.json").read_text())
        depth = np.load(root / "depth_m.npy").astype(np.float64)
        camera = fixed_camera_model(eye=tuple(cam["eye_m"]), target=tuple(cam["target_m"]), resolution=(1280, 720))
        t_wh = np.asarray(fk["T_WH_4x4"])
        t_hp = np.linalg.inv(t_wh) @ plug
        truth6 = np.concatenate((t_hp[:3,3], Rotation.from_matrix(t_hp[:3,:3]).as_euler("xyz")))
        twp = t_wh @ pose_matrix(truth6)
        shell = shell25j_plug_cad_profile(feature_set="shell_only")
        cad = shell.plug_mating
        xyz = cad.xyz[cad.label == 1]
        world = twp[:3,:3] @ xyz.T + twp[:3,3].reshape(3,1)
        uv, pred = project(camera, world.T)
        u = np.rint(uv[:,0]).astype(np.int64); v = np.rint(uv[:,1]).astype(np.int64)
        ok = (pred > 0.03) & (u >= 0) & (u < 1280) & (v >= 0) & (v < 720)
        obs = depth[v[ok], u[ok]]
        print(f"=== {vid} ===  eye={np.round(cam['eye_m'],3)} target={np.round(cam['target_m'],3)}")
        print(f"  shell-band(visible check): pred med={np.median(pred[ok]):.3f} obs med={np.median(obs):.3f} obs p10/p90={np.percentile(obs,10):.3f}/{np.percentile(obs,90):.3f}")
        # what IS the observed surface?  back-project median depth along center ray
        d = np.asarray(cam["target_m"]) - np.asarray(cam["eye_m"]); d /= np.linalg.norm(d)
        center_pt = np.asarray(cam["eye_m"]) + np.median(obs) * d
        print(f"  back-projected median-depth point (world): {np.round(center_pt,3)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
