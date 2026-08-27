"""Hash-bound inner gripping surfaces for the connector-task hand variant."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from kcg_connector.grasp.robust.object_model import file_sha256, load_stl_mesh


_SCHEMA = "carts_task_grip_surface_v2"
_HAND_VARIANT = "CONNECTOR_GRASP_NO_NAIL"
_SEMANTIC = "TASK_GRIP_SURFACE"
_EXPECTED = {
    "finger_1_pad": ("finger_1_task_grip_surface", "f1", "f1Link3"),
    "finger_2_pad": ("finger_2_task_grip_surface", "f2", "f2Link2"),
    "finger_3_pad": ("finger_3_task_grip_surface", "f3", "f3Link3"),
}


def _readonly(value, dtype, *, ndim: int) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ValueError("task-grip surface array is malformed")
    result = np.array(result, copy=True)
    result.setflags(write=False)
    return result


def _repository_file(root: Path, text: object, label: str) -> Path:
    if not isinstance(text, str) or not text or "\\" in text:
        raise ValueError(f"{label} must be a repository-relative POSIX path")
    pure = PurePosixPath(text)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise ValueError(f"{label} must be a normalized repository-relative path")
    path = (root / Path(*pure.parts)).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} resolves outside the repository") from exc
    if not path.is_file():
        raise ValueError(f"{label} is not a regular file")
    return path


@dataclass(frozen=True)
class TaskGripSurface:
    """One exact allowed hand-side surface in its terminal-link frame."""

    surface_name: str
    pad_name: str
    finger_name: str
    link_name: str
    points_local_m: np.ndarray
    faces: np.ndarray
    source_face_indices: np.ndarray
    face_normals_local: np.ndarray
    patch_indices: np.ndarray
    legacy_blue_pad_face_mask: np.ndarray
    source_mesh_path: Path
    source_mesh_sha256: str

    def __post_init__(self) -> None:
        points = _readonly(self.points_local_m, np.float64, ndim=2)
        faces = _readonly(self.faces, np.int64, ndim=2)
        source = _readonly(self.source_face_indices, np.int64, ndim=1)
        normals = _readonly(self.face_normals_local, np.float64, ndim=2)
        patches = _readonly(self.patch_indices, np.int64, ndim=1)
        legacy = _readonly(self.legacy_blue_pad_face_mask, np.bool_, ndim=1)
        count = len(faces)
        if (
            points.shape[1:] != (3,)
            or faces.shape[1:] != (3,)
            or normals.shape != (count, 3)
            or source.shape != (count,)
            or patches.shape != (count,)
            or legacy.shape != (count,)
            or count == 0
            or np.any(faces < 0)
            or np.any(faces >= len(points))
            or np.any(source < 0)
            or len(np.unique(source)) != count
            or np.any(patches < 0)
        ):
            raise ValueError(f"task-grip arrays are inconsistent for {self.link_name}")
        lengths = np.linalg.norm(normals, axis=1)
        if not np.allclose(lengths, 1.0, rtol=0.0, atol=1.0e-10):
            raise ValueError(f"task-grip normals are not unit length for {self.link_name}")
        for name, value in (
            ("points_local_m", points), ("faces", faces),
            ("source_face_indices", source), ("face_normals_local", normals),
            ("patch_indices", patches), ("legacy_blue_pad_face_mask", legacy),
        ):
            object.__setattr__(self, name, value)

    @property
    def triangles_local_m(self) -> np.ndarray:
        result = self.points_local_m[self.faces]
        result.setflags(write=False)
        return result

    @property
    def workspace_reference_center_local_m(self) -> np.ndarray:
        """Area-weighted legacy-PAD center used only as a hand-workspace reference."""

        triangles = self.triangles_local_m[self.legacy_blue_pad_face_mask]
        if len(triangles) == 0:
            raise ValueError(f"{self.link_name} has no legacy-PAD workspace reference")
        areas = 0.5 * np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0],
                     triangles[:, 2] - triangles[:, 0]), axis=1,
        )
        if not np.isfinite(areas.sum()) or areas.sum() <= 0.0:
            raise ValueError(f"{self.link_name} workspace reference area is invalid")
        center = np.average(np.mean(triangles, axis=1), axis=0, weights=areas)
        center.setflags(write=False)
        return center


def _load_row(root: Path, row: Mapping[str, object]) -> TaskGripSurface:
    pad_name = str(row.get("pad_name", ""))
    expected = _EXPECTED.get(pad_name)
    if expected is None or (
        row.get("surface_name"), row.get("finger_name"), row.get("link_name")
    ) != expected:
        raise ValueError("task-grip finger identity changed")
    source_path = _repository_file(root, row.get("source_mesh"), "source_mesh")
    source_hash = str(row.get("source_mesh_sha256", ""))
    arrays_path = _repository_file(root, row.get("surface_npz"), "surface_npz")
    if file_sha256(source_path) != source_hash or file_sha256(arrays_path) != row.get(
        "surface_npz_sha256"
    ):
        raise ValueError(f"task-grip source hash changed for {expected[2]}")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        required = {
            "points_local_m", "faces", "source_face_indices",
            "face_normals_local", "patch_indices", "legacy_blue_pad_face_mask",
        }
        if set(arrays.files) != required:
            raise ValueError(f"task-grip NPZ schema changed for {expected[2]}")
        surface = TaskGripSurface(
            surface_name=expected[0], pad_name=pad_name, finger_name=expected[1],
            link_name=expected[2], points_local_m=arrays["points_local_m"],
            faces=arrays["faces"], source_face_indices=arrays["source_face_indices"],
            face_normals_local=arrays["face_normals_local"],
            patch_indices=arrays["patch_indices"],
            legacy_blue_pad_face_mask=arrays["legacy_blue_pad_face_mask"],
            source_mesh_path=source_path, source_mesh_sha256=source_hash,
        )
    mesh, provenance = load_stl_mesh(source_path, unit="m", orient_outward=False)
    if (
        provenance.source_sha256 != source_hash
        or len(mesh.faces) != int(row.get("source_face_count", -1))
        or np.any(surface.source_face_indices >= len(mesh.faces))
        or not np.array_equal(
            mesh.face_vertices_m[surface.source_face_indices], surface.triangles_local_m
        )
        or not np.allclose(
            mesh.face_normals[surface.source_face_indices],
            surface.face_normals_local, rtol=0.0, atol=1.0e-12,
        )
    ):
        raise ValueError(f"task-grip source-face lineage changed for {expected[2]}")
    if (
        len(surface.faces) != int(row.get("task_face_count", -1))
        or len(np.unique(surface.patch_indices)) != int(row.get("patch_count", -1))
        or int(np.count_nonzero(surface.legacy_blue_pad_face_mask))
        != int(row.get("legacy_blue_pad_face_count", -1))
    ):
        raise ValueError(f"task-grip manifest counts changed for {expected[2]}")
    return surface


def load_task_grip_surfaces(
    repository_root: Path | str,
    manifest_path: Path | str,
    expected_sha256: str,
) -> Mapping[str, TaskGripSurface]:
    """Load the three offline-only semantic surfaces and verify all lineage."""

    root = Path(repository_root).resolve(strict=True)
    supplied = Path(manifest_path)
    path = supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()
    if not path.is_file() or file_sha256(path) != expected_sha256:
        raise ValueError("task-grip manifest is unavailable or its hash changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != _SCHEMA
        or value.get("hand_variant") != _HAND_VARIANT
        or value.get("semantic") != _SEMANTIC
        or value.get("hardware_authorized") is not False
        or value.get("online_control_use_allowed") is not False
        or value.get("object_specific_selection_used") is not False
    ):
        raise ValueError("task-grip manifest safety identity changed")
    rows = value.get("links")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("task-grip manifest must contain exactly three links")
    result = {surface.pad_name: surface for surface in (_load_row(root, row) for row in rows)}
    if set(result) != set(_EXPECTED):
        raise ValueError("task-grip surface coverage changed")
    return MappingProxyType(result)


def bind_task_hand_variant(
    root: Path,
    settings: Mapping[str, Any],
    collision_triangles: Mapping[str, np.ndarray],
) -> tuple[Mapping[str, TaskGripSurface] | None, Mapping[str, np.ndarray], str]:
    """Replace only the three terminal collision meshes for the selected variant."""

    variant = str(settings.get("hand_variant", "LEGACY_NAIL_PRESENT"))
    if variant == "LEGACY_NAIL_PRESENT":
        return None, collision_triangles, variant
    if variant != _HAND_VARIANT:
        raise ValueError(f"unknown hand variant {variant!r}")
    surfaces = load_task_grip_surfaces(
        root,
        settings.get("task_grip_surface_manifest", ""),
        str(settings.get("task_grip_surface_manifest_sha256", "")),
    )
    result = dict(collision_triangles)
    for surface in surfaces.values():
        mesh, provenance = load_stl_mesh(
            surface.source_mesh_path, unit="m", orient_outward=False
        )
        if provenance.source_sha256 != surface.source_mesh_sha256:
            raise ValueError(f"terminal mesh changed for {surface.link_name}")
        triangles = np.array(mesh.face_vertices_m, dtype=np.float64, copy=True)
        triangles.setflags(write=False)
        result[surface.link_name] = triangles
    return surfaces, MappingProxyType(result), variant


def task_noncontact_triangles(
    registered: Mapping[str, np.ndarray],
    surfaces: Mapping[str, TaskGripSurface],
) -> dict[str, np.ndarray]:
    """Subtract exact allowed source faces; all remaining hand faces stay forbidden."""

    result = dict(registered)
    for surface in surfaces.values():
        triangles = np.asarray(registered[surface.link_name])
        if (
            np.any(surface.source_face_indices >= len(triangles))
            or not np.array_equal(
                triangles[surface.source_face_indices], surface.triangles_local_m
            )
        ):
            raise ValueError(f"task surface no longer matches {surface.link_name}")
        keep = np.ones(len(triangles), dtype=np.bool_)
        keep[surface.source_face_indices] = False
        result[surface.link_name] = triangles[keep]
    return result


def motion_faces_current_workspace(
    surfaces: Mapping[str, TaskGripSurface], hand_model, joint_positions_rad,
    base_transform: np.ndarray, pad_name: str, surface_point: np.ndarray,
    surface_normal: np.ndarray, closing_motion: np.ndarray,
    minimum_motion_m_per_phase: float,
) -> bool:
    """Check one face against the actual hand workspace and closing direction."""

    transforms = hand_model.forward_kinematics(
        joint_positions_rad, base_transform=base_transform
    )
    centers = []
    for surface in surfaces.values():
        transform = transforms[surface.link_name]
        local = surface.workspace_reference_center_local_m
        centers.append(local @ transform[:3, :3].T + transform[:3, 3])
    inward = np.mean(centers, axis=0) - np.asarray(surface_point)
    normal = np.asarray(surface_normal)
    motion = np.asarray(closing_motion)
    scale = max(1.0, float(np.linalg.norm(inward) * np.linalg.norm(motion)))
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    return bool(
        np.dot(normal, inward) > tolerance
        and np.dot(motion, inward) > tolerance
        and np.dot(normal, motion) >= float(minimum_motion_m_per_phase)
    )


def allowed_object_grasp_center_m(inputs) -> np.ndarray:
    """Area-weighted center of the currently registered allowed grasp band."""

    mesh = inputs.object_contract.model.mesh
    allowed = np.asarray(inputs.face_roles.face_is_allowed, dtype=np.bool_)
    triangles = np.asarray(mesh.face_vertices_m, dtype=np.float64)[allowed]
    areas = np.asarray(mesh.face_areas_m2, dtype=np.float64)[allowed]
    if (len(triangles) == 0 or not np.all(np.isfinite(areas))
            or float(np.sum(areas)) <= 0.0):
        raise ValueError("allowed object grasp band has no finite area")
    center = np.average(np.mean(triangles, axis=1), axis=0, weights=areas)
    center.setflags(write=False)
    return center


def motion_compatible_with_object_witness(
    surface_points_m: np.ndarray,
    object_points_m: np.ndarray,
    hand_normals_m: np.ndarray,
    object_normals_m: np.ndarray,
    closing_motion_m_per_phase: np.ndarray,
    object_grasp_center_m: np.ndarray,
    minimum_motion_m_per_phase: float,
) -> np.ndarray:
    """Check real face normals and closing motion against the current object band."""

    arrays = [np.asarray(value, dtype=np.float64) for value in (
        surface_points_m, object_points_m, hand_normals_m,
        object_normals_m, closing_motion_m_per_phase)]
    if (any(value.ndim != 2 or value.shape[1:] != (3,) for value in arrays)
            or len({len(value) for value in arrays}) != 1
            or not all(np.all(np.isfinite(value)) for value in arrays)):
        raise ValueError("contact witnesses must be equal finite (N,3) arrays")
    center = np.asarray(object_grasp_center_m, dtype=np.float64)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("object grasp center must be one finite 3-vector")
    hand_points, object_points, hand_normals, object_normals, motion = arrays
    gap = object_points - hand_points
    toward_center = center[None, :] - hand_points
    scale = np.maximum(1.0, np.linalg.norm(toward_center, axis=1)
                       * np.linalg.norm(motion, axis=1))
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    inward_motion = -np.einsum("ij,ij->i", motion, object_normals)
    return (
        (np.einsum("ij,ij->i", hand_normals, toward_center) > tolerance)
        & (np.einsum("ij,ij->i", motion, toward_center) > tolerance)
        & (np.einsum("ij,ij->i", hand_normals, object_normals) < -tolerance)
        & (np.einsum("ij,ij->i", motion, gap) > tolerance)
        & (inward_motion >= float(minimum_motion_m_per_phase))
    )


__all__ = [
    "TaskGripSurface",
    "allowed_object_grasp_center_m",
    "bind_task_hand_variant",
    "load_task_grip_surfaces",
    "motion_faces_current_workspace",
    "motion_compatible_with_object_witness",
    "task_noncontact_triangles",
]
