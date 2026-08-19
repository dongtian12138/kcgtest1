#!/usr/bin/env python3
"""Robust fixed-q7 IK for the raised transport waypoint via scipy least_squares."""
import sys
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/home/noob/WorkPlace/kcgtest1/src/kcg_connector")
from kcg_connector.d38999_tabletop_pick import iiwa14_grasp_tcp_transform

old_arm = np.array([0.435799782, 0.522940416, -0.257136665, -0.686362024, 0.136796539, 1.943438486, 0.650482794])
target_pos = np.array([0.550, 0.185, 0.42])
limits = [(-2.967, 2.967), (-2.094, 2.094), (-2.967, 2.967), (-2.094, 2.094), (-2.967, 2.967), (-2.094, 2.094)]

def fk(q):
    return np.asarray(iiwa14_grasp_tcp_transform(tuple(float(v) for v in q)))

def residual(q6):
    q = np.concatenate((q6, [old_arm[6]]))
    t = fk(q)
    pos = t[:3, 3] - target_pos
    axis = t[:3, 2] - np.array([0.0, 0.0, -1.0])
    return np.concatenate((pos, axis))

lower = [l[0] for l in limits]
upper = [l[1] for l in limits]
result = least_squares(residual, old_arm[:6], bounds=(lower, upper), max_nfev=2000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
q6 = result.x
q = np.concatenate((q6, [old_arm[6]]))
t = fk(q)
print("residual pos:", np.linalg.norm(t[:3,3]-target_pos), "axis:", np.linalg.norm(t[:3,2]-[0,0,-1]))
print("final FK:", np.round(t[:3,3], 6))
print("final arm:", [round(float(v), 9) for v in q])
print("in bounds:", all(l[0]-1e-9 <= v <= l[1]+1e-9 for v, l in zip(q6, limits)))
