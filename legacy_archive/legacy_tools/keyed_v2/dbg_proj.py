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
    root = BASE / "phase7_palm_batch_v1/formal_views/PALM_H0_K1"
    cam = json.loads((root / "camera.json").read_text())
    fk = json.loads((root / "fk.json").read_text())
    depth = np.load(root / "depth_m.npy").astype(np.float64)
    camera = fixed_camera_model(eye=tuple(cam["eye_m"]), target=tuple(cam["target_m"]), resolution=(1280, 720))
    print("camera eye", cam["eye_m"], "target", cam["target_m"])
    print("world_to_camera rows:")
    for row in camera.world_to_camera:
        print("  ", [round(v,4) for v in row])
    t_wh = np.asarray(fk["T_WH_4x4"])
    t_hp = np.linalg.inv(t_wh) @ plug
    truth6 = np.concatenate((t_hp[:3,3], Rotation.from_matrix(t_hp[:3,:3]).as_euler("xyz")))
    twp = t_wh @ pose_matrix(truth6)
    print("twp:")
    print(np.round(twp, 4))
    print("truth6:", np.round(truth6, 4))
    shell = shell25j_plug_cad_profile(feature_set="shell_only")
    xyz = shell.plug_mating.xyz[shell.plug_mating.label == 1]
    print("CAD local xyz z range:", xyz[:,2].min().round(4), xyz[:,2].max().round(4), "shape", xyz.shape)
    print("CAD local xyz x/y range:", xyz[:,0].min().round(4), xyz[:,0].max().round(4), xyz[:,1].min().round(4), xyz[:,1].max().round(4))
    world = twp[:3,:3] @ xyz.T + twp[:3,3].reshape(3,1)
    print("manual first 5:", np.round(world[:, :5].T, 4))
    print("manual z = twp[:3,3][2] + (xyz @ twp[:3,:3].T)[:,2] first5:", np.round((xyz @ twp[:3,:3].T)[:5,2] + twp[2,3], 4))
    print("CAD world z range:", world[:,2].min().round(4), world[:,2].max().round(4))
    uv, pred = project(camera, world.T)
    print("uv range:", uv[:,0].min().round(1), uv[:,0].max().round(1), uv[:,1].min().round(1), uv[:,1].max().round(1))
    print("pred depth range:", pred.min().round(4), pred.max().round(4))
    ok = (pred > 0.03) & (uv[:,0] >= 0) & (uv[:,0] < 1280) & (uv[:,1] >= 0) & (uv[:,1] < 720)
    u = np.rint(uv[ok,0]).astype(np.int64); v = np.rint(uv[ok,1]).astype(np.int64)
    obs = depth[v, u]
    print("obs range:", obs.min().round(4), obs.max().round(4))
    xs = (uv[ok,0] - 640) / 1466.0 * obs
    ys = (uv[ok,1] - 360) / 1466.0 * obs
    print("xs range:", xs.min().round(4), xs.max().round(4), " ys:", ys.min().round(4), ys.max().round(4))
    return 0

if __name__ == "__main__":
    sys.exit(main())
