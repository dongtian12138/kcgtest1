#!/usr/bin/env python3
"""Per-point depth discrepancy diagnostic for C1 at the posthoc truth pose."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/home/noob/WorkPlace/kcgtest1/src/kcg_connector")
from kcg_connector.d38999_cad_registration import (
    proxy_cad_points, shell25j_plug_cad_profile, fixed_camera_model, project,
)
from kcg_connector.d38999_inhand_multiview import pose_matrix

BASE = Path("/home/noob/WorkPlace/kcgtest1/artifacts/kcg_connector/"
            "d38999_postgrasp_visual_ft_e2e_v1")
ROOT = BASE / "phase6_t_hp_h0_palm_c1_v1/seed000/formal_views/PALM_H0"
SNAP = BASE / "phase1_snapshot_gate_v3/seed000/postgrasp_snapshot_gate/snapshot_gate.json"


def main() -> int:
    depth = np.load(ROOT / "depth_m.npy").astype(np.float64)
    cam = json.loads((ROOT / "camera.json").read_text())
    camera = fixed_camera_model(eye=tuple(cam["eye_m"]), target=tuple(cam["target_m"]),
                                resolution=(1280, 720))
    fk = json.loads((ROOT / "fk.json").read_text())
    t_wh = np.asarray(fk["T_WH_4x4"])
    snap = json.loads(SNAP.read_text())
    q = np.asarray(snap["plug_root_state"]["orientation_wxyz"])
    rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])
    plug = np.eye(4); plug[:3,:3] = rot.as_matrix()
    plug[:3,3] = np.asarray(snap["plug_root_state"]["position_m"])
    t_hp = np.linalg.inv(t_wh) @ plug
    t_wp = t_wh @ t_hp
    truth6 = np.concatenate((t_hp[:3,3], Rotation.from_matrix(t_hp[:3,:3]).as_euler("xyz")))
    twp = t_wh @ pose_matrix(truth6)
    print("T_WP (truth):")
    print(np.round(twp, 4))

    legacy_plug, _ = proxy_cad_points()
    shell = shell25j_plug_cad_profile(feature_set="shell_plus_socket")

    for name, cad in (("legacy", legacy_plug), ("shell25j", shell.plug_mating)):
        sel = cad.label == 1
        xyz = cad.xyz[sel]
        world = twp[:3,:3] @ xyz.T + twp[:3,3].reshape(3,1)
        uv, pred = project(camera, world.T)
        u = np.rint(uv[:,0]).astype(np.int64); v = np.rint(uv[:,1]).astype(np.int64)
        ok = (pred > 0.03) & (u >= 0) & (u < 1280) & (v >= 0) & (v < 720)
        obs = depth[v[ok], u[ok]]
        dvalid = np.isfinite(obs) & (obs > 0)
        diff = (pred[ok][dvalid] - obs[dvalid]) * 1000.0
        print(f"{name}: n={np.sum(ok)} depth_valid={np.sum(dvalid)}")
        print(f"  predicted depth: min={pred[ok][dvalid].min():.3f} med={np.median(pred[ok][dvalid]):.3f} max={pred[ok][dvalid].max():.3f}")
        print(f"  observed depth:  min={obs[dvalid].min():.3f} med={np.median(obs[dvalid]):.3f} max={obs[dvalid].max():.3f}")
        print(f"  diff pred-obs mm: p10={np.percentile(diff,10):.1f} med={np.median(diff):.1f} p90={np.percentile(diff,90):.1f}")
        print(f"  |diff|>2mm: {np.mean(np.abs(diff) > 2):.2f}  |diff|>5mm: {np.mean(np.abs(diff) > 5):.2f}")
        # where do they project in image
        print(f"  u range [{u[ok].min()},{u[ok].max()}] v range [{v[ok].min()},{v[ok].max()}]")
    return 0

if __name__ == "__main__":
    sys.exit(main())
