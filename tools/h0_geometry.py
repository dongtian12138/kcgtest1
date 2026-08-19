#!/usr/bin/env python3
"""Compute H0 geometry: plug face pose in world, hand frames, grip interval."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SNAP = Path("/home/noob/WorkPlace/kcgtest1/artifacts/kcg_connector/"
            "d38999_postgrasp_visual_ft_e2e_v1/phase1_snapshot_gate_v3/seed000/"
            "postgrasp_snapshot_gate/snapshot_gate.json")
FK = Path("/home/noob/WorkPlace/kcgtest1/artifacts/kcg_connector/"
          "d38999_postgrasp_visual_ft_e2e_v1/phase6_t_hp_h0_palm_capture_v1/"
          "seed000/formal_views/PALM_H0/fk.json")


def main() -> int:
    snap = json.loads(SNAP.read_text())
    fk = json.loads(FK.read_text())
    plug_pos = np.asarray(snap["plug_root_state"]["position_m"])
    quat_wxyz = np.asarray(snap["plug_root_state"]["orientation_wxyz"])
    rot = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    rmat = rot.as_matrix()
    print("plug world position :", plug_pos.round(4))
    print("plug world R        :")
    print(rmat.round(4))
    z_local = rmat @ np.asarray([0.0, 0.0, 1.0])
    print("plug local +z in world (face outward per CAD):", z_local.round(4))
    # face plane center = plug position (face at local z=0)
    face_center = plug_pos
    print("mating face center (world):", face_center.round(4))
    print("plug axis tilt from world z (deg):",
          float(np.degrees(np.arccos(np.clip(z_local[2], -1, 1)))).__round__(3))
    # hand frames
    t_wh = np.asarray(fk["T_WH_4x4"])
    tcp = np.asarray(fk["tcp_pose_4x4"])
    print("handbase world:", t_wh[:3, 3].round(4), " TCP world:", tcp[:3, 3].round(4))
    hand_z = t_wh[:3, 2]
    print("hand z axis (world):", hand_z.round(4))
    grip = [0.41748, 0.44148]
    for g in grip:
        print(f"grip_local_z {g} -> world point:", (t_wh[:3,3] + g*hand_z).round(4))
    # face center in hand frame (T_HP truth, posthoc)
    t_hp = np.linalg.inv(t_wh) @ np.eye(4) if False else None
    plug_world = np.eye(4); plug_world[:3,:3] = rmat; plug_world[:3,3] = plug_pos
    t_hp = np.linalg.inv(t_wh) @ plug_world
    print("posthoc T_HP translation (hand frame, m):", t_hp[:3,3].round(4))
    print("posthoc T_HP rotvec (rad):", Rotation.from_matrix(t_hp[:3,:3]).as_rotvec().round(4))
    print("posthoc T_HP euler xyz (deg):", np.degrees(Rotation.from_matrix(t_hp[:3,:3]).as_euler('xyz')).round(3))
    # nominal hand->plug from pick config (what control believes)
    pick_origin = np.asarray([0.520, -0.210, 0.200])
    nominal_plug = np.eye(4); nominal_plug[:3,3] = pick_origin
    nominal_tcp = np.asarray(fk["tcp_pose_4x4"])  # H0 tcp
    tcp_from_handbase = np.eye(4); tcp_from_handbase[2,3] = -0.4
    nominal_hand = nominal_tcp @ tcp_from_handbase
    nom_t_hp = np.linalg.inv(nominal_hand) @ nominal_plug
    print("nominal T_HP translation:", nom_t_hp[:3,3].round(4))
    return 0

if __name__ == "__main__":
    sys.exit(main())
