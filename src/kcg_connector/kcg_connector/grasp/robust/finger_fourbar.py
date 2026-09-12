"""Measured planar finger closure, in the original URDF hinge coordinates.

The two serial hinges stay in the robot tree. The physical connecting rod
imposes |R(q)(b + R(p)c) - a| = length. Its signed assembly branch is part of
the geometry contract; a nearest-point choice must not switch that branch.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np


def _rotate(angle: float, point: np.ndarray) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c*point[0]-s*point[1], s*point[0]+c*point[1]])


@dataclass(frozen=True)
class FingerFourBar:
    source_joint: str
    follower_joint: str
    base_anchor_xy_m: tuple[float, float]
    distal_joint_xy_m: tuple[float, float]
    distal_anchor_xy_m: tuple[float, float]
    rod_length_m: float
    branch_sign: int

    def __post_init__(self):
        values = (*self.base_anchor_xy_m, *self.distal_joint_xy_m,
                  *self.distal_anchor_xy_m, self.rod_length_m)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Four-bar geometry must be finite")
        if self.branch_sign not in (-1, 1) or self.rod_length_m <= 0:
            raise ValueError("A positive rod length and signed branch are required")
        if not self.source_joint or not self.follower_joint or self.source_joint == self.follower_joint:
            raise ValueError("Distinct source and follower hinges are required")
        if min(np.linalg.norm(self.distal_joint_xy_m), np.linalg.norm(self.distal_anchor_xy_m)) <= 0:
            raise ValueError("The two moving links must have nonzero lengths")

    def position_and_derivative(self, source_angle: float) -> tuple[float, float]:
        q = float(source_angle)
        if not math.isfinite(q):
            raise ValueError("Four-bar source angle must be finite")
        a = np.asarray(self.base_anchor_xy_m)
        b = _rotate(q, np.asarray(self.distal_joint_xy_m))
        c_local = np.asarray(self.distal_anchor_xy_m)
        radius = float(np.linalg.norm(c_local))
        difference = a-b
        separation = float(np.linalg.norm(difference))
        if separation <= 1e-12:
            raise ValueError("Coincident four-bar circle centers")
        direction = difference/separation
        along = (radius*radius-self.rod_length_m**2+separation**2)/(2*separation)
        height_squared = radius*radius-along*along
        if height_squared <= 1e-18:
            raise ValueError("Four-bar pose is infeasible or at a toggle")
        c = b+along*direction+self.branch_sign*math.sqrt(height_squared)*np.array([-direction[1], direction[0]])
        bc = c-b
        raw = math.atan2(bc[1],bc[0])-q-math.atan2(c_local[1],c_local[0])
        p = math.atan2(math.sin(raw), math.cos(raw))
        rod = c-a
        denominator = float(rod @ np.array([-bc[1],bc[0]]))
        if abs(denominator) <= 1e-12:
            raise ValueError("Four-bar velocity map is singular")
        derivative = -float(rod @ np.array([-c[1],c[0]]))/denominator
        return p, derivative

    def closure_error(self, source_angle: float, follower_angle: float) -> float:
        c = _rotate(source_angle, np.asarray(self.distal_joint_xy_m)+
                    _rotate(follower_angle, np.asarray(self.distal_anchor_xy_m)))
        return float(np.linalg.norm(c-np.asarray(self.base_anchor_xy_m))-self.rod_length_m)

    def source_interval(self, source_bounds: tuple[float,float], follower_bounds: tuple[float,float]) -> tuple[float,float]:
        """Intersect limits on the selected monotone, non-toggle work branch.

        The measured hand operates on a monotone branch. This checks that
        premise over its declared interval and solves boundary angles, rather
        than retaining the old affine follower limit transformation.
        """
        lo, hi = map(float, source_bounds)
        flo, fhi = map(float, follower_bounds)
        if not lo < hi or flo > fhi:
            raise ValueError("Invalid hinge limits")
        probe = [self.position_and_derivative(float(q)) for q in np.linspace(lo,hi,129)]
        slopes = np.asarray([r[1] for r in probe])
        if not (np.all(slopes > 0) or np.all(slopes < 0)):
            raise ValueError("The declared four-bar range is not a single monotone branch")
        increasing = bool(slopes[0] > 0)
        p0, p1 = probe[0][0],probe[-1][0]
        if max(p0,p1) < flo or min(p0,p1) > fhi:
            raise ValueError("No four-bar angles satisfy both hinge limits")

        def inverse(value):
            left,right=lo,hi
            for _ in range(60):
                middle=(left+right)/2
                p,_=self.position_and_derivative(middle)
                if (p < value) == increasing:left=middle
                else:right=middle
            return (left+right)/2

        if increasing:
            return (lo if p0>=flo else inverse(flo),hi if p1<=fhi else inverse(fhi))
        return (lo if p0<=fhi else inverse(fhi),hi if p1>=flo else inverse(flo))

    @classmethod
    def from_mapping(cls, row: Mapping):
        return cls(source_joint=str(row["source_joint"]),follower_joint=str(row["follower_joint"]),
                   base_anchor_xy_m=tuple(row["base_anchor_xy_m"]),
                   distal_joint_xy_m=tuple(row["distal_joint_xy_m"]),
                   distal_anchor_xy_m=tuple(row["distal_anchor_xy_m"]),
                   rod_length_m=float(row["rod_length_m"]),branch_sign=int(row["branch_sign"]))


def load_finger_fourbars(path):
    document = json.loads(Path(path).read_text())
    if document.get("schema") != "kcg.hand.fourbar.v1":
        raise ValueError("Unsupported measured hand mechanism contract")
    couplings = {row["follower_joint"]: FingerFourBar.from_mapping(row)
                 for row in document["finger_joints"].values()}
    if set(couplings) != {"f1j3", "f2j2", "f3j3"}:
        raise ValueError("The measured contract must cover all three distal joints")
    return document, couplings
