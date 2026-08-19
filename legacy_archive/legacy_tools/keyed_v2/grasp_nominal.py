#!/usr/bin/env python3
import sys
sys.path.insert(0, "/home/noob/WorkPlace/kcgtest1/src/kcg_connector")
import numpy as np
from kcg_connector.d38999_tabletop_pick import load_d38999_tabletop_pick_config, iiwa14_grasp_tcp_transform

pick = load_d38999_tabletop_pick_config("/home/noob/WorkPlace/kcgtest1/src/kcg_connector/config/d38999_tabletop_pick_v1.yaml")
print("grasp_arm_rad:", pick.motion.grasp_arm_rad)
tcp = np.asarray(iiwa14_grasp_tcp_transform(tuple(pick.motion.grasp_arm_rad)))
print("grasp TCP world:")
print(tcp.round(4))
tcp_from_handbase = np.eye(4); tcp_from_handbase[2,3] = -float(pick.geometry_candidate.handbase_to_tcp_m)
hand = tcp @ tcp_from_handbase
print("grasp handbase world:", hand[:3,3].round(4), " z:", hand[:3,2].round(4))
grip = [0.41748, 0.44148]
for g in grip:
    p = hand[:3,3] + g*hand[:3,2]
    print("grip_local_z", g, "-> world", p.round(4))
lo = np.asarray(pick.geometry_candidate.loose_settled_origin_m)
print("loose_settled_origin:", lo)
nom_plug = np.eye(4); nom_plug[:3,3] = lo
nom_hp = np.linalg.inv(hand) @ nom_plug
print("nominal T_HP at grasp (hand frame):", nom_hp[:3,3].round(4))
print("grasp candidate fields:")
for k, v in pick.geometry_candidate.__dict__.items():
    print("  ", k, "=", v)
