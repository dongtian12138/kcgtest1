#!/usr/bin/env python3
"""Joint palm(K1)+wrist(H0) T_HP estimation.  POSTHOC diagnostic only."""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/home/noob/WorkPlace/kcgtest1/src/kcg_connector")
from kcg_connector.d38999_cad_registration import (
    CameraModel, CadPoints, fixed_camera_model, project,
    shell25j_plug_cad_profile,
)
from kcg_connector.d38999_inhand_multiview import pose_matrix
from kcg_connector.postgrasp_shadow_estimator import (
    FormalView, estimate_postgrasp_T_HP,
)

BASE = Path("/home/noob/WorkPlace/kcgtest1/artifacts/kcg_connector/"
            "d38999_postgrasp_visual_ft_e2e_v1")
SNAP = BASE / "phase1_snapshot_gate_v3/seed000/postgrasp_snapshot_gate/snapshot_gate.json"
PALM = BASE / "phase7_palm_wrist_joint_v2/formal_views/PALM_H0_K0"
WRIST = BASE / "phase7_palm_wrist_joint_v2/formal_views/WRIST_H0"
SIDECAR = BASE / "phase7_palm_wrist_joint_v2/posthoc_truth_sidecar.json"


def load_view(root: Path, view_id: str, group: str) -> FormalView:
    vd = root
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


def cad_with_nut():
    prof = shell25j_plug_cad_profile(feature_set="shell_plus_socket")
    mating = prof.plug_mating
    occl = prof.plug_occluders
    xyz = np.concatenate([mating.xyz, occl.xyz], axis=0)
    normal = np.concatenate([mating.normal, occl.normal], axis=0)
    label = np.concatenate([mating.label, occl.label], axis=0)
    edge = np.concatenate([mating.edge, occl.edge], axis=0)
    return CadPoints(xyz, normal, label, edge)


def active_depth_rms(view, twp, cad, camera):
    """Active (visible-point) depth residual RMS in mm at a given world pose."""
    sel = cad.label == 1
    xyz = cad.xyz[sel]
    world = twp[:3,:3] @ xyz.T + twp[:3,3].reshape(3,1)
    uv, pred = project(camera, world.T)
    u = np.rint(uv[:,0]).astype(np.int64); v = np.rint(uv[:,1]).astype(np.int64)
    ok = (pred > 0.03) & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
    obs = view.depth[v[ok], u[ok]]
    dvalid = np.isfinite(obs) & (obs > 0.0)
    visible = dvalid & (pred[ok] <= obs + 0.0015)
    diff = (pred[ok][visible] - obs[visible]) * 1000.0
    return float(np.sqrt(np.mean(diff**2))) if len(diff) else float("nan"), int(np.sum(visible)), int(np.sum(ok))


def main() -> int:
    snap = json.loads(SNAP.read_text())
    q = np.asarray(snap["plug_root_state"]["orientation_wxyz"])
    rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])
    plug = np.eye(4); plug[:3,:3] = rot.as_matrix()
    plug[:3,3] = np.asarray(snap["plug_root_state"]["position_m"])
    fk = json.loads((PALM / "fk.json").read_text())
    t_wh = np.asarray(fk["T_WH_4x4"])
    t_hp_snapshot = np.linalg.inv(t_wh) @ plug
    sidecar = json.loads(SIDECAR.read_text())
    qs = np.asarray(sidecar["plug_orientation_wxyz"])
    rots = Rotation.from_quat([qs[1], qs[2], qs[3], qs[0]])
    plugs = np.eye(4); plugs[:3,:3] = rots.as_matrix()
    plugs[:3,3] = np.asarray(sidecar["plug_position_m"])
    t_hp = np.linalg.inv(t_wh) @ plugs
    snap_truth6 = np.concatenate((t_hp_snapshot[:3,3],
                                  Rotation.from_matrix(t_hp_snapshot[:3,:3]).as_euler("xyz")))
    truth6 = np.concatenate((t_hp[:3,3], Rotation.from_matrix(t_hp[:3,:3]).as_euler("xyz")))
    axis_true = t_hp[:3,:3] @ np.array([0,0,1.0]); axis_true /= np.linalg.norm(axis_true)
    print("truth(capture-time) xyz(mm):", (truth6[:3]*1000).round(2), "axis:", axis_true.round(4))
    print("truth(snapshot)      xyz(mm):", (snap_truth6[:3]*1000).round(2),
          " drift_from_snapshot(mm):", (np.linalg.norm(truth6[:3]-snap_truth6[:3])*1000).round(2))
    initial = np.zeros(12)
    initial[:6] = np.array([0.0, 0.0, 0.4485, math.pi, 0.0, 0.0])
    palm = load_view(PALM, "PALM_H0_K1", "postgrasp_inhand_views")
    wrist = load_view(WRIST, "WRIST_H0", "postgrasp_second_inhand_camera_views")
    cad = cad_with_nut()
    configs = [
        ("joint_palm_mating_only", None, (1,), dict(plug_feature_set="mating_only",
             occlusion_policy="ignore_foreground_occluded", edge_policy="depth_gated",
             optimizer_variant="multistart_physical_jacobian", multistart_count=17)),
        ("joint_mating_plus_nut_rear", cad, (1, 2, 5), dict(plug_feature_set="mating_plus_nut_body",
             occlusion_policy="ignore_foreground_occluded", edge_policy="depth_gated",
             optimizer_variant="multistart_physical_jacobian", multistart_count=17)),
    ]
    for label, custom_cad, labels, kwargs in configs:
        if custom_cad is not None:
            from kcg_connector.d38999_cad_registration import PLUG_MATING, PLUG_NUT_BODY, PLUG_REAR_BODY
            kwargs = dict(kwargs)
            # estimator derives plug_labels from plug_feature_set; mating_plus_nut_body -> (PLUG_MATING, PLUG_NUT_BODY)
            result = estimate_postgrasp_T_HP(
                [palm, wrist], initial, plug_cad=custom_cad,
                receptacle_cad=shell25j_plug_cad_profile().receptacle, **kwargs)
        else:
            result = estimate_postgrasp_T_HP([palm, wrist], initial, **kwargs)
        h0, h1 = result["c2"]["hypotheses"]
        e0 = np.asarray(h0["T_hand_plug_xyz_rpy"])
        err = min(np.linalg.norm(e0[:3]-truth6[:3])*1000,
                  np.linalg.norm(np.asarray(h1["T_hand_plug_xyz_rpy"])[:3]-truth6[:3])*1000)
        r0 = pose_matrix(e0); axis_est = r0[:3,:3] @ np.array([0,0,1.0]); axis_est /= np.linalg.norm(axis_est)
        axis_err = float(np.degrees(np.arccos(np.clip(abs(float(axis_est @ axis_true)), -1, 1))))
        sup = h0["plug_support_gate_failed"] or h1["plug_support_gate_failed"]
        print(f"{label}: err_mm={err:.2f} axis_err_deg={axis_err:.2f} "
              f"rms={min(h0['residual_rms'], h1['residual_rms']):.3f} support_fail={sup} "
              f"cond0={h0['condition_number'] and round(h0['condition_number'],1)}")
        for d in (h0.get("plug_support_diagnostics") or []):
            print(f"    {d['view_id']}: in_frame={d['in_frame_fraction']:.2f} vis={d['visible_depth_support_fraction']:.2f} "
                  f"fg_occ={d['foreground_occluded_fraction']:.2f} edge={d['edge_support_fraction']:.2f} dgate={d['depth_gated_edge_support_fraction']:.2f}")
        print(f"    YAW_0 xyz(mm)={ (e0[:3]*1000).round(2)} rpy(deg)={np.degrees(e0[3:]).round(2)}")
        from kcg_connector.d38999_cad_registration import proxy_cad_points
        shell = shell25j_plug_cad_profile(feature_set="shell_only")
        twp_est = palm.T_WH @ pose_matrix(e0)
        for vname, vv in (("PALM", palm), ("WRIST", wrist)):
            rms_mm, n_vis, n_ok = active_depth_rms(vv, twp_est, shell.plug_mating, vv.camera)
            print(f"    {vname} active depth RMS at estimate: {rms_mm:.2f} mm (visible {n_vis}/{n_ok})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
