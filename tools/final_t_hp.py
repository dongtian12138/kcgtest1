#!/usr/bin/env python3
"""Final T_HP estimation on the best palm view (K1) with several configs."""
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
            "d38999_postgrasp_visual_ft_e2e_v1")
SNAP = BASE / "phase1_snapshot_gate_v3/seed000/postgrasp_snapshot_gate/snapshot_gate.json"
VIEW = BASE / "phase7_palm_batch_v1/formal_views/PALM_H0_K1"


def load_view() -> FormalView:
    vd = VIEW
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
    return FormalView(view_id="PALM_H0_K1", timestamp_utc="2026-08-15T00:00:00Z",
                      rgb=rgb, depth=depth, camera=camera,
                      T_WH=np.asarray(fk["T_WH_4x4"]),
                      T_WC=np.asarray(fk["T_WC_4x4"]),
                      group="postgrasp_inhand_views",
                      extrinsic_source="T_HC_calibrated")


def main() -> int:
    snap = json.loads(SNAP.read_text())
    q = np.asarray(snap["plug_root_state"]["orientation_wxyz"])
    rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])
    plug = np.eye(4); plug[:3,:3] = rot.as_matrix()
    plug[:3,3] = np.asarray(snap["plug_root_state"]["position_m"])
    fk = json.loads((VIEW / "fk.json").read_text())
    t_wh = np.asarray(fk["T_WH_4x4"])
    t_hp = np.linalg.inv(t_wh) @ plug
    truth6 = np.concatenate((t_hp[:3,3], Rotation.from_matrix(t_hp[:3,:3]).as_euler("xyz")))
    print("posthoc truth T_HP xyz(mm):", (truth6[:3]*1000).round(2),
          "rpy(deg):", np.degrees(truth6[3:]).round(2))
    initial = np.zeros(12)
    initial[:6] = np.array([0.0, 0.0, 0.4485, math.pi, 0.0, 0.0])
    view = load_view()
    configs = [
        ("legacy_ignore_dgate_ms9", dict(cad_profile="legacy_axisymmetric",
             occlusion_policy="ignore_foreground_occluded", edge_policy="depth_gated",
             optimizer_variant="multistart_physical_jacobian", multistart_count=9)),
        ("legacy_ignore_dgate_ms17", dict(cad_profile="legacy_axisymmetric",
             occlusion_policy="ignore_foreground_occluded", edge_policy="depth_gated",
             optimizer_variant="multistart_physical_jacobian", multistart_count=17)),
        ("legacy_ignore_dgate10mm_ms17", dict(cad_profile="legacy_axisymmetric",
             occlusion_policy="ignore_foreground_occluded", edge_policy="depth_gated",
             edge_depth_band_m=0.010,
             optimizer_variant="multistart_physical_jacobian", multistart_count=17)),
        ("shell25j_ignore_dgate_ms17", dict(cad_profile="shell25j_c2_visible",
             occlusion_policy="ignore_foreground_occluded", edge_policy="depth_gated",
             optimizer_variant="multistart_physical_jacobian", multistart_count=17)),
    ]
    best = None
    for label, kwargs in configs:
        result = estimate_postgrasp_T_HP([view], initial, **kwargs)
        h0, h1 = result["c2"]["hypotheses"]
        e0 = np.asarray(h0["T_hand_plug_xyz_rpy"]); e1 = np.asarray(h1["T_hand_plug_xyz_rpy"])
        err = min(np.linalg.norm(e0[:3]-truth6[:3])*1000, np.linalg.norm(e1[:3]-truth6[:3])*1000)
        rms = min(h0["residual_rms"], h1["residual_rms"])
        sup = h0["plug_support_gate_failed"] or h1["plug_support_gate_failed"]
        print(f"{label}: err_mm={err:.2f} rms={rms:.3f} support_fail={sup} "
              f"cond0={h0['condition_number'] and round(h0['condition_number'],1)}")
        key = err if not sup else err + 500
        if best is None or key < best[0]:
            best = (key, label, result, e0, e1)
    _, label, result, e0, e1 = best
    print()
    print("=== FINAL ===", label)
    inv = result["c2_invariant_5dof"]
    axis_true = t_hp[:3,:3] @ np.array([0,0,1.0]); axis_true /= np.linalg.norm(axis_true)
    print("YAW_0  T_HP xyz(mm):", (e0[:3]*1000).round(2), " rpy(deg):", np.degrees(e0[3:]).round(2))
    print("YAW_PI T_HP xyz(mm):", (e1[:3]*1000).round(2), " rpy(deg):", np.degrees(e1[3:]).round(2))
    print("C2-invariant 5DOF xyz(mm):", (np.asarray(inv[:3])*1000).round(2),
          "axis:", np.round(inv[3:], 4))
    print("truth xyz(mm):", (truth6[:3]*1000).round(2), " axis:", np.round(axis_true, 4))
    print("grasp offset vs nominal(0,0,448.5): est dz(mm) =", ((e0[2]-0.4485)*1000).round(2),
          " truth dz(mm) =", ((truth6[2]-0.4485)*1000).round(2))
    print("residual_rms:", [h["residual_rms"] for h in result["c2"]["hypotheses"]])
    print("pose_valid:", result["pose_valid"], result["pose_valid_reasons"])
    out = BASE / "phase7_palm_batch_v1/t_hp_final_estimate.json"
    out.write_text(json.dumps({
        "schema_version": "kcg_d38999_palm_t_hp_v1",
        "role": "posthoc_diagnostic_estimate",
        "formal_estimator_input": False,
        "control_authorized": False,
        "view_id": "PALM_H0_K1",
        "palm_eye_plug_m": [0.045, 0.0, 0.100],
        "config": label,
        "initial_T_HP_xyz_rpy": initial[:6].tolist(),
        "YAW_0_T_hand_plug_xyz_rpy": e0.tolist(),
        "YAW_PI_T_hand_plug_xyz_rpy": e1.tolist(),
        "c2_invariant_5dof": inv,
        "posthoc_truth_T_hand_plug_xyz_rpy": truth6.tolist(),
        "translation_error_vs_snapshot_mm": float(np.linalg.norm(e0[:3]-truth6[:3])*1000),
        "pose_valid": result["pose_valid"],
        "pose_valid_reasons": result["pose_valid_reasons"],
        "residual_rms": [h["residual_rms"] for h in result["c2"]["hypotheses"]],
        "threshold_label": "SIM_TUNING_ONLY_CANDIDATE",
    }, indent=2) + "\n")
    print("written:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
