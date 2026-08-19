#!/usr/bin/env python3
"""CPU harness: run estimate_postgrasp_T_HP on the phase6 palm C1 archive
across CAD profile / occlusion / edge / optimizer variants and compare
against posthoc snapshot truth (diagnostic only)."""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/home/noob/WorkPlace/kcgtest1/src/kcg_connector")
from kcg_connector.d38999_cad_registration import CameraModel, fixed_camera_model
from kcg_connector.d38999_inhand_multiview import matrix_pose, pose_matrix
from kcg_connector.postgrasp_shadow_estimator import (
    FormalView,
    estimate_postgrasp_T_HP,
)

BASE = Path("/home/noob/WorkPlace/kcgtest1/artifacts/kcg_connector/"
            "d38999_postgrasp_visual_ft_e2e_v1")
SNAP = BASE / "phase1_snapshot_gate_v3/seed000/postgrasp_snapshot_gate/snapshot_gate.json"
FK = None  # loaded per view


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
    t_wh = np.asarray(fk["T_WH_4x4"])
    t_hp = np.linalg.inv(t_wh) @ plug
    return t_hp


def pose6_of(t: np.ndarray) -> np.ndarray:
    return np.concatenate((t[:3,3], Rotation.from_matrix(t[:3,:3]).as_euler("xyz")))


def report(label, result, truth6):
    print(f"--- {label} ---")
    print(f"  success={result['success']} status={result['status']}")
    print(f"  reject_reason={result.get('reject_reason')}")
    for hyp in result["c2"]["hypotheses"]:
        est = np.asarray(hyp["T_hand_plug_xyz_rpy"])
        tm = np.linalg.norm(est[:3] - truth6[:3]) * 1000.0
        # compare rotation allowing C2 partner: use axis alignment
        r_est = pose_matrix(est); r_true = pose_matrix(truth6)
        axis_est = r_est[:3,:3] @ np.array([0,0,1.0]); axis_true = r_true[:3,:3] @ np.array([0,0,1.0])
        axis_err = float(np.degrees(np.arccos(np.clip(abs(np.dot(axis_est, axis_true)), -1, 1))))
        print(f"  {hyp['id']}: xyz={est[:3].round(4)} rpy_deg={np.degrees(est[3:]).round(2)} "
              f"|err_mm={tm:.2f} axis_err_deg={axis_err:.2f} "
              f"resid_rms={hyp['residual_rms']:.3f} cond={hyp['condition_number'] if hyp['condition_number'] is None else round(hyp['condition_number'],1)} "
              f"support_fail={hyp['plug_support_gate_failed']} solver={hyp['solver_status']}")
        for d in hyp.get("plug_support_diagnostics", []):
            print(f"    support: {d['view_id']} in_frame={d['in_frame_fraction']:.3f} "
                  f"vis={d['visible_depth_support_fraction']:.3f} fg_occ={d['foreground_occluded_fraction']:.3f} "
                  f"edge={d['edge_support_fraction']:.3f} dgate_edge={d['depth_gated_edge_support_fraction']:.3f}")
    print(f"  pose_valid={result['pose_valid']} reasons={result['pose_valid_reasons']}")
    print(f"  c2_invariant_5dof={[round(v,4) for v in result['c2_invariant_5dof']]}")


def truth_pose_diagnostics(view, truth6, **kwargs):
    from kcg_connector.postgrasp_shadow_estimator import _ResidualProblem
    from kcg_connector.d38999_cad_registration import proxy_cad_points, shell25j_plug_cad_profile
    plug_occluder = None
    if kwargs.get("cad_profile") == "shell25j_c2_visible":
        prof = shell25j_plug_cad_profile(feature_set=kwargs.get("cad_profile_feature_set", "shell_plus_socket"))
        plug, rec = prof.plug_mating, prof.receptacle
        plug_occluder = prof.plug_occluders
    else:
        plug, rec = proxy_cad_points()
    state = np.zeros(12); state[:6] = truth6
    problem = _ResidualProblem(
        [view], plug, rec, state, frozen_mask=(False,)*6+(True,)*6,
        endpoints=("plug",), plug_occluder_cad=plug_occluder,
        occlusion_policy=kwargs.get("occlusion_policy", "baseline"),
        edge_policy=kwargs.get("edge_policy", "global"),
    )
    residual = problem.residual(state)
    rms = float(np.sqrt(np.mean(residual ** 2)))
    print(f"  [truth-pose diagnostic] rms={rms:.3f} support:")
    for d in problem.last_plug_support:
        print(f"    {d['view_id']} in_frame={d['in_frame_fraction']:.3f} vis={d['visible_depth_support_fraction']:.3f} fg_occ={d['foreground_occluded_fraction']:.3f} edge={d['edge_support_fraction']:.3f}")
    return rms


def main() -> int:
    root = BASE / "phase6_t_hp_h0_palm_c1_v1/seed000/formal_views"
    view = load_view(root, "PALM_H0")
    snap = json.loads(SNAP.read_text())
    fk = json.loads((root / "PALM_H0/fk.json").read_text())
    t_hp = truth_t_hp(snap, fk)
    truth6 = pose6_of(t_hp)
    print("posthoc truth T_HP xyz_rpy:", np.concatenate((truth6[:3].round(4), np.degrees(truth6[3:]).round(3))))
    initial = np.zeros(12)
    initial[:6] = np.array([0.0, 0.0, 0.4485, math.pi, 0.0, 0.0])

    configs = [
        ("legacy_default",
         dict()),
        ("legacy_ignore_dgate_ms",
         dict(cad_profile="legacy_axisymmetric",
              occlusion_policy="ignore_foreground_occluded",
              edge_policy="depth_gated",
              optimizer_variant="multistart_physical_jacobian",
              multistart_count=9)),
        ("shell25j_default",
         dict(cad_profile="shell25j_c2_visible")),
        ("shell25j_ignore_dgate_ms",
         dict(cad_profile="shell25j_c2_visible",
              occlusion_policy="ignore_foreground_occluded",
              edge_policy="depth_gated",
              optimizer_variant="multistart_physical_jacobian",
              multistart_count=9)),
        ("shell25j_ignore_cadoccl_dgate_ms",
         dict(cad_profile="shell25j_c2_visible",
              occlusion_policy="ignore_foreground_and_cad_occluder",
              edge_policy="depth_gated",
              optimizer_variant="multistart_physical_jacobian",
              multistart_count=9)),
        ("shell25j_mating_plus_nut_ignore_dgate_ms",
         dict(cad_profile="shell25j_c2_visible",
              plug_feature_set="mating_plus_nut_body",
              occlusion_policy="ignore_foreground_occluded",
              edge_policy="depth_gated",
              optimizer_variant="multistart_physical_jacobian",
              multistart_count=9)),
    ]
    for label, kwargs in configs:
        try:
            print(f"=== {label} ===")
            truth_pose_diagnostics(view, truth6, **kwargs)
            result = estimate_postgrasp_T_HP([view], initial, **kwargs)
            report(label, result, truth6)
        except Exception as exc:  # noqa: BLE001
            print(f"--- {label} FAILED: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
