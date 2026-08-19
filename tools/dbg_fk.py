import json
import math
import sys
from pathlib import Path

import numpy as np
import trimesh
import xml.etree.ElementTree as ET

BASE = Path("/home/noob/WorkPlace/kcgtest1")
URDF = BASE / "artifacts/kcg_connector/urdf/handarm.urdf"
MESH = BASE / "src/iiwa_description/meshes/hand"
SNAP = BASE / ("artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1/"
               "phase1_snapshot_gate_v3/seed000/postgrasp_snapshot_gate/snapshot_gate.json")
FK = BASE / ("artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1/"
             "phase7_palm_batch_v1/formal_views/PALM_H0_K0/fk.json")


def rpy_m(rpy):
    r, p, y = (float(v) for v in rpy.split())
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cp*cy, sr*sp*cy-cr*sy, cr*sp*cy+sr*sy],
        [cp*sy, sr*sp*sy+cr*cy, cr*sp*sy-sr*cy],
        [-sp, sr*cp, cr*cp],
    ])


def main() -> int:
    root = ET.parse(URDF).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    snap = json.loads(SNAP.read_text())
    q = snap["robot_state"]["q_rad"]
    order = ["f1j1", "f1j2", "f1j3", "f2j1", "f2j2", "f3j1", "f3j2", "f3j3"]
    hand_q = dict(zip(order, q[7:15]))
    fk = json.loads(FK.read_text())
    t_wh = np.asarray(fk["T_WH_4x4"])

    link_pose = {"handbase_link": t_wh}
    parent_of = {j.get("name"): j.find("parent").get("link") for j in root.findall("joint")}
    child_of = {j.get("name"): j.find("child").get("link") for j in root.findall("joint")}
    for jn in order:
        j = joints[jn]
        o = j.find("origin")
        xyz = np.array([float(v) for v in o.get("xyz").split()])
        m = np.eye(4)
        m[:3, :3] = rpy_m(o.get("rpy"))
        m[:3, 3] = xyz
        axis_text = j.find("axis").get("xyz")
        if jn == "f1j3":
            print("DEBUG f1j3 axis repr:", repr(axis_text))
        ax = np.array([float(v) for v in axis_text.split()])
        c, s = math.cos(hand_q[jn]), math.sin(hand_q[jn])
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        rot = np.eye(4)
        rot[:3, :3] = np.eye(3) + s*K + (1-c)*(K @ K)
        link_pose[child_of[jn]] = link_pose[parent_of[jn]] @ m @ rot

    plug_xy = np.array([0.5198, -0.2099])
    for name in ["f1Link1", "f1Link2", "f1Link3", "f2Link1", "f2Link2",
                 "f3Link1", "f3Link2", "f3Link3"]:
        mesh = trimesh.load(MESH / f"{name}.STL")
        mesh.apply_transform(link_pose[name])
        b = mesh.bounds
        c = mesh.center_mass
        r = np.linalg.norm(c[:2] - plug_xy) * 1000.0
        print(f"{name:10s} center_world={np.round(c,4)} z=({b[0,2]:.4f},{b[1,2]:.4f}) r={r:.1f}mm")
    print("plug face world z=0.252; nut z=0.266..0.283 r 19.5..24mm; fingers should grip there")
    return 0


if __name__ == "__main__":
    sys.exit(main())
