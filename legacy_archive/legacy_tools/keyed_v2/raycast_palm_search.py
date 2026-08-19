#!/usr/bin/env python3
"""Raycast search for palm camera poses with clear view of the Plug mating face.

FK the hand URDF at the snapshot finger angles, merge hand + plug geometry,
and score candidate camera poses by the fraction of face-rays that reach the
face disk unobstructed.  Validated against the empirical K0-K4/C1 captures.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import trimesh
import xml.etree.ElementTree as ET

BASE = Path("/home/noob/WorkPlace/kcgtest1")
URDF = BASE / "artifacts/kcg_connector/urdf/handarm.urdf"
MESH_DIR = BASE / "src/iiwa_description/meshes/hand"
SNAP = BASE / ("artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1/"
               "phase1_snapshot_gate_v3/seed000/postgrasp_snapshot_gate/snapshot_gate.json")
FK = BASE / ("artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1/"
             "phase7_palm_batch_v1/formal_views/PALM_H0_K0/fk.json")

HAND_JOINT_ORDER = ["f1j1", "f1j2", "f1j3", "f2j1", "f2j2", "f3j1", "f3j2", "f3j3"]
HAND_LINKS = ["handbase_link", "f1Link1", "f1Link2", "f1Link3",
              "f2Link1", "f2Link2", "f3Link1", "f3Link2", "f3Link3"]


def rpy_matrix(rpy):
    r, p, y = (float(v) for v in rpy.split())
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def joint_transform(origin, axis, angle):
    xyz = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
    rpy = origin.get("rpy", "0 0 0")
    m = np.eye(4)
    m[:3, :3] = rpy_matrix(rpy)
    m[:3, 3] = xyz
    a = np.array([float(v) for v in axis.split()])
    rot = np.eye(4)
    c, s = math.cos(angle), math.sin(angle)
    ax = a / (np.linalg.norm(a) + 1e-12)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    rot[:3, :3] = np.eye(3) + s * K + (1 - c) * (K @ K)
    return m @ rot


def main() -> int:
    root = ET.parse(URDF).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    snap = json.loads(SNAP.read_text())
    q = snap["robot_state"]["q_rad"]
    hand_q = dict(zip(HAND_JOINT_ORDER, q[7:15]))
    fk = json.loads(FK.read_text())
    t_wh = np.asarray(fk["T_WH_4x4"])

    # FK hand links (ignore mimic tags; snapshot holds actual joint values)
    link_pose = {"handbase_link": t_wh}
    parent_of = {j.get("name"): j.find("parent").get("link") for j in root.findall("joint")}
    child_of = {j.get("name"): j.find("child").get("link") for j in root.findall("joint")}
    for jname in HAND_JOINT_ORDER:
        j = joints[jname]
        child = child_of[jname]
        parent = parent_of[jname]
        if child == "grasp_tcp":
            continue
        origin = j.find("origin")
        axis = j.find("axis").get("xyz")
        t_parent = link_pose[parent]
        link_pose[child] = t_parent @ joint_transform(origin, axis, hand_q[jname])

    scene = []
    for link in HAND_LINKS:
        mesh = trimesh.load(MESH_DIR / f"{link}.STL")
        mesh.apply_transform(link_pose[link])
        scene.append(mesh)
    hand = trimesh.util.concatenate(scene)

    # Plug geometry (asset dims): face disk, shell band, nut, rear body
    plug_q = np.asarray(snap["plug_root_state"]["orientation_wxyz"])
    rot = trimesh.transformations.quaternion_matrix(
        [plug_q[1], plug_q[2], plug_q[3], plug_q[0]])
    t_wp = rot.copy()
    t_wp[:3, 3] = np.asarray(snap["plug_root_state"]["position_m"])
    plug_parts = []
    face = trimesh.creation.cylinder(radius=0.0155, height=0.001, sections=64)
    face.apply_translation([0, 0, 0.0005])
    plug_parts.append(face)
    shell = trimesh.creation.annulus(r_min=0.0165, r_max=0.0190, height=0.010, sections=64)
    shell.apply_translation([0, 0, 0.005])
    plug_parts.append(shell)
    nut = trimesh.creation.annulus(r_min=0.0195, r_max=0.0240, height=0.017, sections=64)
    nut.apply_translation([0, 0, 0.0225])
    plug_parts.append(nut)
    rear = trimesh.creation.cylinder(radius=0.02215, height=0.014, sections=64)
    rear.apply_translation([0, 0, 0.024])
    plug_parts.append(rear)
    plug = trimesh.util.concatenate(plug_parts)
    plug.apply_transform(t_wp)

    world = trimesh.util.concatenate([hand, plug])
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(world)

    # Face targets in plug frame: center + rings
    targets = [[0.0, 0.0, 0.0005]]
    for radius in (7.0, 15.0):
        for k in range(16):
            a = 2 * math.pi * k / 16
            targets.append([radius * math.cos(a), radius * math.sin(a), 0.0005])
    targets = np.asarray(targets) * 1e-3
    t_pw = np.linalg.inv(t_wp)

    def score(eye_plug):
        eye = np.asarray(eye_plug) * 1e-3
        eye_w = (t_wp[:3, :3] @ eye) + t_wp[:3, 3]
        origins = np.tile(eye_w, (len(targets), 1))
        dirs = (t_wp[:3, :3] @ targets.T).T + t_wp[:3, 3] - origins
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        hits, _, _ = intersector.intersects_location(origins, dirs)
        if len(hits) == 0:
            return 0.0
        hit_plug = (hits - t_wp[:3, 3]) @ t_wp[:3, :3]
        r = np.linalg.norm(hit_plug[:, :2], axis=1)
        z = hit_plug[:, 2]
        face_hits = int(np.sum((r <= 0.0156) & (z >= -0.0002) & (z <= 0.0012)))
        return face_hits / len(targets)

    known = {
        "PALM_H0_K0": (0.030, 0.0, 0.120),
        "PALM_H0_K1": (0.045, 0.0, 0.100),
        "PALM_H0_K2": (0.020, 0.0, 0.140),
        "PALM_H0_K3": (-0.008, 0.039, 0.120),
        "PALM_H0_K4": (0.038, -0.014, 0.120),
        "C1_REF": (0.070, 0.0, 0.100),
    }
    print("=== model validation vs empirical ===")
    for name, eye in known.items():
        s = score(eye)
        x, y, z = eye
        r = math.hypot(x, y) * 1e3
        az = math.degrees(math.atan2(y, x))
        dist = math.hypot(math.hypot(x, y), z) * 1e3
        px = 31.0 / (dist * 1e-3) * 1466
        print(f"{name:12s} eye=(x={x:+.3f},y={y:+.3f},z={z:.3f}) r={r:5.1f}mm az={az:+6.1f}deg "
              f"dist={dist:5.1f}mm face_px={px:5.0f} -> score={s:.3f}")

    print()
    print("=== grid search ===")
    results = []
    for az in range(0, 360, 10):
        a = math.radians(az)
        for r in (30.0, 40.0, 50.0, 60.0, 70.0):
            for z in (80.0, 100.0, 120.0, 150.0, 200.0, 250.0, 300.0):
                x, y = r * math.cos(a) * 1e-3, r * math.sin(a) * 1e-3
                eye = (x, y, z * 1e-3)
                s = score(eye)
                dist = math.hypot(math.hypot(x, y), z * 1e-3) * 1e3
                px = 31.0 / (dist * 1e-3) * 1466
                if s > 0.8:
                    results.append((s, px, az, r, z))
    results.sort(reverse=True)
    print(f"candidates with score>0.8: {len(results)}")
    for s, px, az, r, z in results[:25]:
        a = math.radians(az)
        x = r * math.cos(a) * 1e-3
        y = r * math.sin(a) * 1e-3
        print(f"  score={s:.2f} face_px={px:5.0f} eye=(x={x:+.3f},y={y:+.3f},z={z/1000:.3f}) "
              f"r={r:4.0f}mm az={az:+4d}deg z={z:4.0f}mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
