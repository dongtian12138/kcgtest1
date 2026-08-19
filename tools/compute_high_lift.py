#!/usr/bin/env python3
"""Solve the high-lift waypoint: plug vertical, TCP at (0.55, 0.185, 0.45), q7 fixed."""
import sys
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, "/home/noob/WorkPlace/kcgtest1/src/kcg_connector")
from kcg_connector.d38999_tabletop_pick import iiwa14_grasp_tcp_transform

q7 = 0.650482794
grasp_arm = np.array([-0.129948040338, 0.415863528857, -0.404365705983, -1.108380667037, 0.160080395004, 1.646562177393, -0.107043622479])
target_pos = np.array([0.550, 0.185, 0.45])
limits = [(-2.967, 2.967), (-2.094, 2.094), (-2.967, 2.967), (-2.094, 2.094), (-2.967, 2.967), (-2.094, 2.094)]
grasp_tcp = np.asarray(iiwa14_grasp_tcp_transform(tuple(float(v) for v in grasp_arm)))
target_rot = grasp_tcp[:3,:3].copy()

def fk(q):
    return np.asarray(iiwa14_grasp_tcp_transform(tuple(float(v) for v in q)))

def residual(q6):
    q = np.concatenate((q6, [q7]))
    t = fk(q)
    return np.concatenate((t[:3,3]-target_pos, (t[:3,:3]-target_rot).ravel()))

seed = np.concatenate((grasp_arm[:6], ))
result = least_squares(residual, seed, bounds=([l[0] for l in limits],[l[1] for l in limits]), max_nfev=4000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
q = np.concatenate((result.x, [q7]))
t = fk(q)
print("pos err:", np.linalg.norm(t[:3,3]-target_pos), "rot err:", np.linalg.norm(t[:3,:3]-target_rot))
print("FK:", np.round(t[:3,3],6), "axis:", np.round(t[:3,2],4))
print("arm:", [round(float(v),9) for v in q])
