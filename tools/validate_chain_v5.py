#!/usr/bin/env python3
"""Validate v5 chain T_HP/T_RP against capture-time and end-of-chain truth."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/home/noob/WorkPlace/kcgtest1/src/kcg_connector")
from kcg_connector.d38999_inhand_multiview import matrix_pose, pose_matrix

BASE = Path("/home/noob/WorkPlace/kcgtest1/artifacts/kcg_connector/"
            "d38999_postgrasp_visual_ft_e2e_v1/phase8_visual_chain_v5")
RECEPTACLE_POS = np.array([0.550, 0.185, 0.2615])
RECEPTACLE_AXIS = np.array([0.0, 0.0, 1.0])


def main() -> int:
    report = json.loads((BASE / "visual_chain_report.json").read_text())
    side = json.loads((BASE / "posthoc_truth_sidecar.json").read_text())
    qs = np.asarray(side["plug_orientation_wxyz"])
    rot = Rotation.from_quat([qs[1], qs[2], qs[3], qs[0]])
    plug_h0 = np.eye(4); plug_h0[:3,:3] = rot.as_matrix()
    plug_h0[:3,3] = np.asarray(side["plug_position_m"])
    palm_fk = json.loads((BASE / "formal_views/PALM_H0_K0/fk.json").read_text())
    t_wh_h0 = np.asarray(palm_fk["T_WH_4x4"])
    t_hp_truth = np.linalg.inv(t_wh_h0) @ plug_h0
    truth6 = np.concatenate((t_hp_truth[:3,3],
                             Rotation.from_matrix(t_hp_truth[:3,:3]).as_euler("xyz")))
    est0 = np.asarray(report["stage_t_hp"]["c2_hypotheses"][0]["T_hand_plug_xyz_rpy"])
    est1 = np.asarray(report["stage_t_hp"]["c2_hypotheses"][1]["T_hand_plug_xyz_rpy"])
    err0 = np.linalg.norm(est0[:3]-truth6[:3])*1000
    err1 = np.linalg.norm(est1[:3]-truth6[:3])*1000
    axis_t = t_hp_truth[:3,:3] @ np.array([0,0,1.0]); axis_t/=np.linalg.norm(axis_t)
    axis_e = pose_matrix(est0)[:3,:3] @ np.array([0,0,1.0]); axis_e/=np.linalg.norm(axis_e)
    print(f"T_HP truth xyz(mm): {(truth6[:3]*1000).round(2)} tilt {np.degrees(np.arccos(abs(axis_t[2]))):.2f} deg")
    print(f"T_HP YAW_0  err={err0:.2f}mm  tilt {np.degrees(np.arccos(abs(axis_e[2]))):.2f} deg")
    print(f"T_HP YAW_PI err={err1:.2f}mm")
    # T_RP truth with end-of-chain plug pose
    side_end = json.loads((BASE / "posthoc_truth_sidecar_end.json").read_text())
    qe = np.asarray(side_end["plug_orientation_wxyz"])
    rote = Rotation.from_quat([qe[1], qe[2], qe[3], qe[0]])
    plug_end = np.eye(4); plug_end[:3,:3] = rote.as_matrix()
    plug_end[:3,3] = np.asarray(side_end["plug_position_m"])
    wr0_fk = json.loads((BASE / "formal_views/W_R0/fk.json").read_text())
    t_wh_w = np.asarray(wr0_fk["T_WH_4x4"])
    t_wp = t_wh_w @ np.linalg.inv(t_wh_h0) @ plug_end  # plug pose in W_R0 hand frame chain
    # simpler: plug world pose at end is plug_end; T_WP at W_R0 = t_wh_w @ inv(t_wh_h0) @ plug_end
    t_wp = t_wh_w @ (np.linalg.inv(t_wh_h0) @ plug_end)
    ref = np.array([1.0,0.0,0.0]) - np.dot(np.array([1.0,0.0,0.0]), RECEPTACLE_AXIS)*RECEPTACLE_AXIS
    ref /= np.linalg.norm(ref)
    t_wr = np.eye(4)
    t_wr[:3,0] = ref; t_wr[:3,1] = np.cross(RECEPTACLE_AXIS, ref); t_wr[:3,2] = RECEPTACLE_AXIS
    t_wr[:3,3] = RECEPTACLE_POS
    t_rp_truth = np.linalg.inv(t_wp) @ t_wr
    rp_truth6 = matrix_pose(t_rp_truth)
    rp_est = np.asarray(report["stage_t_rp"]["T_receptacle_plug_xyz_rpy"])
    print(f"T_RP truth xyz(mm): {(rp_truth6[:3]*1000).round(2)}")
    print(f"T_RP est   xyz(mm): {(rp_est[:3]*1000).round(2)}")
    print(f"T_RP translation err (end-of-chain plug, approx): {np.linalg.norm(rp_est[:3]-rp_truth6[:3])*1000:.2f} mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
