"""Fixed hand-occluder CAD for the post-grasp estimator.

The three-finger hand is held at the frozen finger command throughout the
post-grasp captures.  Its geometry in the hand frame is therefore a KNOWN,
a-priori model (robot URDF + frozen joint command), not object truth and not
a contact report.  These points define which face samples can never be
visible from a hand-mounted camera, for the support denominator only.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

import numpy as np

from kcg_connector.d38999_cad_registration import CadPoints


def _load_stl_vertices(path: Path) -> np.ndarray:
    """Minimal STL reader (binary or ASCII) returning (N, 3) vertices."""
    data = path.read_bytes()
    if data[:5].strip() == b"solid" and b"facet normal" in data[:512]:
        # ASCII STL
        vertices = []
        for block in data.split(b"vertex")[1:]:
            values = block.split(b"\n")[0].split()
            if len(values) == 3:
                vertices.append([float(v) for v in values])
        return np.asarray(vertices, dtype=np.float64)
    count = int(np.frombuffer(data[80:84], dtype="<u4")[0])
    records = np.frombuffer(
        data[84 : 84 + count * 50], dtype=np.uint8
    ).reshape(count, 50)
    floats = records[:, :48].copy().view("<f4").reshape(count, 12)
    vertices = floats[:, 3:12].reshape(-1, 3).astype(np.float64)
    return vertices


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-vertex normals via face-normal accumulation."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    accumulated = np.zeros_like(vertices)
    np.add.at(accumulated, faces[:, 0], normals)
    np.add.at(accumulated, faces[:, 1], normals)
    np.add.at(accumulated, faces[:, 2], normals)
    length = np.linalg.norm(accumulated, axis=1, keepdims=True)
    length = np.maximum(length, 1.0e-12)
    return accumulated / length

HAND_JOINT_ORDER = (
    "f1j1", "f1j2", "f1j3", "f2j1", "f2j2", "f3j1", "f3j2", "f3j3",
)
HAND_LINKS = (
    "handbase_link",
    "f1Link1", "f1Link2", "f1Link3",
    "f2Link1", "f2Link2",
    "f3Link1", "f3Link2", "f3Link3",
)
HAND_OCCLUDER_LABEL = 99


def _rpy_matrix(rpy: str) -> np.ndarray:
    r, p, y = (float(value) for value in rpy.split())
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array(
        [
            [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
            [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
            [-sp, sr * cp, cr * cp],
        ]
    )


def _joint_transform(origin: ET.Element, axis: str, angle: float) -> np.ndarray:
    xyz = np.array(
        [float(value) for value in origin.get("xyz", "0 0 0").split()]
    )
    transform = np.eye(4)
    transform[:3, :3] = _rpy_matrix(origin.get("rpy", "0 0 0"))
    transform[:3, 3] = xyz
    a = np.array([float(value) for value in axis.split()])
    a = a / (np.linalg.norm(a) + 1.0e-12)
    c, s = math.cos(angle), math.sin(angle)
    k = np.array(
        [[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]]
    )
    rotation = np.eye(4)
    rotation[:3, :3] = np.eye(3) + s * k + (1.0 - c) * (k @ k)
    return transform @ rotation


def build_hand_occluder_cad(
    hand_q: Sequence[float],
    urdf_path: Path | str,
    mesh_dir: Path | str,
    max_points_per_link: int = 3000,
) -> CadPoints:
    """Sample the hand meshes in the hand frame at the given joint angles.

    ``hand_q`` must follow HAND_JOINT_ORDER (8 values).  The returned points
    are authored in the hand frame, labelled HAND_OCCLUDER_LABEL, and are
    only used as structural occluders (never as fit features).
    """
    values = np.asarray(hand_q, dtype=np.float64)
    if values.shape != (8,) or not np.all(np.isfinite(values)):
        raise ValueError("hand_q must be 8 finite values in HAND_JOINT_ORDER")
    urdf_path = Path(urdf_path)
    mesh_dir = Path(mesh_dir)
    root = ET.parse(urdf_path).getroot()
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    link_pose = {"handbase_link": np.eye(4)}
    parent_of = {
        joint.get("name"): joint.find("parent").get("link")
        for joint in root.findall("joint")
    }
    child_of = {
        joint.get("name"): joint.find("child").get("link")
        for joint in root.findall("joint")
    }
    for index, joint_name in enumerate(HAND_JOINT_ORDER):
        joint = joints[joint_name]
        child = child_of[joint_name]
        if child == "grasp_tcp":
            continue
        link_pose[child] = link_pose[parent_of[joint_name]] @ _joint_transform(
            joint.find("origin"), joint.find("axis").get("xyz"), values[index]
        )
    points = []
    normals = []
    for link in HAND_LINKS:
        link_vertices = _load_stl_vertices(mesh_dir / f"{link}.STL")
        link_normals = _vertex_normals(
            link_vertices, np.arange(len(link_vertices)).reshape(-1, 3)
        )
        link_vertices = (
            link_pose[link][:3, :3] @ link_vertices.T
        ).T + link_pose[link][:3, 3]
        link_normals = link_normals @ link_pose[link][:3, :3].T
        if len(link_vertices) > max_points_per_link:
            keep = np.linspace(
                0, len(link_vertices) - 1, max_points_per_link
            ).astype(int)
            link_vertices = link_vertices[keep]
            link_normals = link_normals[keep]
        points.append(link_vertices.astype(np.float64))
        normals.append(link_normals.astype(np.float64))
    xyz = np.concatenate(points, axis=0)
    normal = np.concatenate(normals, axis=0)
    label = np.full(len(xyz), HAND_OCCLUDER_LABEL, dtype=np.int16)
    edge = np.zeros(len(xyz), dtype=bool)
    return CadPoints(xyz=xyz, normal=normal, label=label, edge=edge)


__all__ = [
    "HAND_JOINT_ORDER",
    "HAND_OCCLUDER_LABEL",
    "build_hand_occluder_cad",
]
