#!/usr/bin/env python3
"""Offline batch analysis of phase7 palm views: face visibility + T_HP estimation.

POSTHOC_TRUTH_ONLY diagnostics; never feeds control."""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/home/noob/WorkPlace/kcgtest1/src/kcg_connector")
from kcg_connector.d38999_cad_registration import (
    CameraModel, fixed_camera_model, shell25j_plug_cad_profile, project,
)
from kcg_connector.d38999_inhand_multiview import pose_matrix
from kcg_connector.postgrasp_shadow_estimator import (
    FormalView, estimate_postgrasp_T_HP,
)

BASE = Path("/home/noob/WorkPlace/kcgtest1/artifacts/kcg_connector/"
            "d38999_postgrasp_visual_ft_e2e_v1")
SNAP = BASE / "phase1_snapshot_gate_v3/seed000/postgrasp_snapshot_gate/snapshot_gate.json"
OUTPUT = BASE / "phase7_palm_batch_v1/formal_views"


def load_view(root: Path, view_id: str) -> FormalView:
    vd = root / view_id
    rgb = cv2.cvtColor(cv2.imread(str(vd / "rgb.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    depth = np.load(vd / "depth_m.npy").astype(np.float32)
    cam = json.loads((vd / "camera.json").read_text())
    eye = tuple(cam["eye_m"]); target = tuple(cam["target_m"])
    model = fixed_camera_model(eye=eye, target=target, resolution=(1280, 720))
    intrinsics = np.asarray(cam["intrinsics"])
    camera = CameraModel(1280, 720, float(intrinsics[0,0]), float(intrinsics[1,1]),
                         float(intrinsics[0,2]), float(intrinsics[1,2]),
                         tuple(model.position_world), tuple(model.world_to_camera))
    fk = json.loads((vd / "fk.json").read_text())
    return FormalView(view_id=view_id, timestamp_utc="2026-08-15T00:00:00Z",
                      rgb=rgb, depth=depth, camera=camera,
                      T_WH=np.asarray(fk["T_WH_4x4"]),
                      T_WC=np.asarray(fk["T_WC_4x4"]),
                      group="postgrasp_inhand_views",
                      extrinsic_source="T_HC_calibrated")


def truth_t_hp(snap: dict, fk: dict) -> np.ndarray:
    q = np.asarray(snap["plug_root_state"]["orientation_wxyz"])
    rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])
    plug = np.eye(4); plug[:3,:3] = rot.as_matrix()
    plug[:3,3] = np.asarray(snap["plug_root_state"]["position_m"])
    return np.linalg.inv(np.asarray(fk["T_WH_4x4"])) @ plug


def face_metrics(view: FormalView, twp: np.ndarray) -> dict:
    shell = shell25j_plug_cad_profile(feature_set="shell_plus_socket")
    cad = shell.plug_mating
    sel = cad.label == 1
    xyz = cad.xyz[sel]
    world = twp[:3,:3] @ xyz.T + twp[:3,3].reshape(3,1)
    uv, pred = project(view.camera, world.T)
    u = uv[:,0]; v = uv[:,1]
    inside = (pred > 0.03) & (u >= 0) & (u < view.camera.width) & (v >= 0) & (v < view.camera.height)
    ui = np.clip(u[inside].astype(np.int64), 0, view.camera.width-1)
    vi = np.clip(v[inside].astype(np.int64), 0, view.camera.height-1)
    obs = view.depth[vi, ui]
    dvalid = np.isfinite(obs) & (obs > 0.0)
    visible = dvalid & (pred[inside] <= obs + 0.0015)
    behind = dvalid & ~visible
    bbox_w = float(np.max(u[inside]) - np.min(u[inside])) if np.sum(inside) else 0.0
    bbox_h = float(np.max(v[inside]) - np.min(v[inside])) if np.sum(inside) else 0.0
    return {
        "projected": int(np.sum(inside)),
        "short_axis_px": min(bbox_w, bbox_h),
        "visible_fraction": float(np.mean(visible)) if np.sum(inside) else 0.0,
        "foreground_occluded_fraction": float(np.mean(behind)) if np.sum(inside) else 0.0,
        "depth_invalid_fraction": float(np.mean(~dvalid)) if np.sum(inside) else 1.0,
    }


def main() -> int:
    snap = json.loads(SNAP.read_text())
    view_ids = sorted(p.name for p in OUTPUT.iterdir() if p.is_dir())
    c1_root = BASE / "phase6_t_hp_h0_palm_c1_v1/seed000/formal_views"
    if c1_root.is_dir():
        view_ids.append("C1_REF")
        OUTPUT_EXTRA = {"C1_REF": c1_root}
    else:
        OUTPUT_EXTRA = {}
    initial = np.zeros(12)
    initial[:6] = np.array([0.0, 0.0, 0.4485, math.pi, 0.0, 0.0])
    results = []
    for view_id in view_ids:
        view_root = OUTPUT_EXTRA.get(view_id, OUTPUT)
        view = load_view(view_root, view_id)
        fk = json.loads((view_root / view_id / "fk.json").read_text())
        t_hp = truth_t_hp(snap, fk)
        truth6 = np.concatenate((t_hp[:3,3], Rotation.from_matrix(t_hp[:3,:3]).as_euler("xyz")))
        twp = view.T_WH @ pose_matrix(truth6)
        fm = face_metrics(view, twp)
        print(f"=== {view_id} === face_metrics={ {k: round(v,3) for k,v in fm.items()} }")
        best = None
        for label, kwargs in (
            ("shell25j_baseline", dict(cad_profile="shell25j_c2_visible")),
            ("shell25j_ignore_dgate_ms", dict(cad_profile="shell25j_c2_visible",
                 occlusion_policy="ignore_foreground_occluded",
                 edge_policy="depth_gated",
                 optimizer_variant="multistart_physical_jacobian", multistart_count=9)),
            ("legacy_ignore_dgate_ms", dict(cad_profile="legacy_axisymmetric",
                 occlusion_policy="ignore_foreground_occluded",
                 edge_policy="depth_gated",
                 optimizer_variant="multistart_physical_jacobian", multistart_count=9)),
        ):
            try:
                result = estimate_postgrasp_T_HP([view], initial, **kwargs)
            except Exception as exc:  # noqa: BLE001
                print(f"   {label}: FAILED {type(exc).__name__}: {exc}")
                continue
            hyp0 = result["c2"]["hypotheses"][0]
            hyp1 = result["c2"]["hypotheses"][1]
            est0 = np.asarray(hyp0["T_hand_plug_xyz_rpy"])
            est1 = np.asarray(hyp1["T_hand_plug_xyz_rpy"])
            err0 = np.linalg.norm(est0[:3] - truth6[:3]) * 1000.0
            err1 = np.linalg.norm(est1[:3] - truth6[:3]) * 1000.0
            err = min(err0, err1)
            rms = min(hyp0["residual_rms"], hyp1["residual_rms"])
            sup_fail = hyp0["plug_support_gate_failed"] or hyp1["plug_support_gate_failed"]
            supp = (hyp0.get("plug_support_diagnostics") or [{}])[0]
            print(f"   {label}: err_mm={err:.2f} rms={rms:.3f} support_fail={sup_fail} "
                  f"vis={supp.get('visible_depth_support_fraction', float('nan')):.3f} "
                  f"fg_occ={supp.get('foreground_occluded_fraction', float('nan')):.3f} "
                  f"dgate_edge={supp.get('depth_gated_edge_support_fraction', float('nan')):.3f}")
            score = err if not sup_fail else err + 500.0
            if best is None or score < best[0]:
                best = (score, label, est0, est1, rms, sup_fail, result)
        if best is not None:
            _, label, est0, est1, rms, sup_fail, result = best
            results.append((view_id, label, est0, est1, err, rms, sup_fail, result))
            print(f"   BEST {label}: { {k: round(v,3) for k,v in fm.items()} }")
    print()
    print("=== ranking ===")
    for view_id, label, est0, est1, err, rms, sup_fail, result in sorted(results, key=lambda r: r[4]):
        print(f"{view_id:16s} {label:28s} err_mm={err:6.2f} rms={rms:6.3f} support_fail={sup_fail}")
    if results:
        view_id, label, est0, est1, err, rms, sup_fail, result = sorted(results, key=lambda r: r[4])[0]
        print()
        print("=== winner ===", view_id, label)
        print(json.dumps({
            "view_id": view_id, "config": label,
            "YAW_0_T_hand_plug_xyz_rpy": est0.tolist(),
            "YAW_PI_T_hand_plug_xyz_rpy": est1.tolist(),
            "posthoc_truth_T_hand_plug_xyz_rpy": np.concatenate((truth6[:3], truth6[3:])).tolist(),
            "translation_error_mm": err,
            "residual_rms": rms,
            "support_gate_failed": sup_fail,
        }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
