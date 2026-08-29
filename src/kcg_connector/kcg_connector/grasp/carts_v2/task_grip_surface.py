"""Hash-bound inner gripping surfaces for the connector-task hand variant."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from kcg_connector.grasp.robust.object_model import file_sha256, load_stl_mesh


_SCHEMA = "carts_task_grip_surface_v2"
_HAND_VARIANT = "CONNECTOR_GRASP_NO_NAIL"
_DIRECT_NAILFREE_VARIANT = "DIRECT_USER_NAILFREE_STL"
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
    if variant == _DIRECT_NAILFREE_VARIANT:
        source = Path(str(settings.get("direct_nailfree_stl", ""))).expanduser()
        if not source.is_absolute():
            source = root / source
        source = source.resolve(strict=True)
        expected_hash = str(settings.get("direct_nailfree_stl_sha256", ""))
        if not source.is_file() or file_sha256(source) != expected_hash:
            raise ValueError("direct nail-free STL is unavailable or changed")
        if settings.get("direct_nailfree_stl_unit") != "mm":
            raise ValueError("direct nail-free STL unit changed")
        import trimesh

        raw_mesh = trimesh.load_mesh(source, force="mesh", process=False)
        raw_vertices = np.asarray(raw_mesh.vertices, dtype=np.float64)
        raw_faces = np.asarray(raw_mesh.faces, dtype=np.int64)
        if (
            raw_vertices.ndim != 2
            or raw_vertices.shape[1:] != (3,)
            or raw_faces.ndim != 2
            or raw_faces.shape[1:] != (3,)
            or len(raw_faces) == 0
            or not np.all(np.isfinite(raw_vertices))
        ):
            raise ValueError("direct nail-free STL mesh is malformed")
        pad_surface = settings.get("direct_nailfree_pad_surface")
        if (
            not isinstance(pad_surface, Mapping)
            or pad_surface.get("semantic_authority")
            != "USER_CONFIRMED_BLUE_PAD_BODY"
            or pad_surface.get("definition")
            != "WHOLE_CONNECTED_PAD_SURFACE_BOUNDED_BY_SHARP_INTERFACES"
        ):
            raise ValueError("direct nail-free full-pad semantics are unavailable")
        source_face_range = np.asarray(
            pad_surface.get("source_face_index_range_zero_based_inclusive", ()),
            dtype=np.int64,
        )
        if (
            source_face_range.shape != (2,)
            or source_face_range[0] < 0
            or source_face_range[1] < source_face_range[0]
            or source_face_range[1] >= len(raw_faces)
        ):
            raise ValueError("direct nail-free full-pad face range is malformed")
        source_faces = np.arange(
            source_face_range[0], source_face_range[1] + 1, dtype=np.int64
        )
        transforms = settings.get("direct_nailfree_link_transforms")
        if not isinstance(transforms, Mapping) or set(transforms) != {
            expected[2] for expected in _EXPECTED.values()
        }:
            raise ValueError("direct nail-free link transforms are incomplete")

        source_triangles = raw_vertices[raw_faces] * 0.001
        source_normals = np.asarray(raw_mesh.face_normals, dtype=np.float64)
        result = dict(collision_triangles)
        surfaces: dict[str, TaskGripSurface] = {}
        for pad_name, (surface_name, finger_name, link_name) in _EXPECTED.items():
            row = transforms[link_name]
            if not isinstance(row, Mapping):
                raise ValueError(f"direct transform for {link_name} is malformed")
            yaw = float(row.get("yaw_rad", float("nan")))
            translation = np.asarray(row.get("translation_m"), dtype=np.float64)
            if not np.isfinite(yaw) or translation.shape != (3,) or not np.all(
                np.isfinite(translation)
            ):
                raise ValueError(f"direct transform for {link_name} is non-finite")
            cosine, sine = np.cos(yaw), np.sin(yaw)
            rotation = np.asarray(
                ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
                dtype=np.float64,
            )
            transformed = source_triangles @ rotation.T + translation
            result[link_name] = transformed
            task_triangles = transformed[source_faces]
            task_normals = source_normals[source_faces] @ rotation.T
            points = task_triangles.reshape(-1, 3)
            faces = np.arange(len(points), dtype=np.int64).reshape(-1, 3)
            surfaces[pad_name] = TaskGripSurface(
                surface_name=surface_name,
                pad_name=pad_name,
                finger_name=finger_name,
                link_name=link_name,
                points_local_m=points,
                faces=faces,
                source_face_indices=source_faces,
                face_normals_local=task_normals,
                patch_indices=np.zeros(len(source_faces), dtype=np.int64),
                legacy_blue_pad_face_mask=np.ones(
                    len(source_faces), dtype=np.bool_
                ),
                source_mesh_path=source,
                source_mesh_sha256=expected_hash,
            )
        return MappingProxyType(surfaces), MappingProxyType(result), variant
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


def generate_axial_pad_intersection_grasp(
    inputs,
    *,
    palm_joint_position_rad: float | None = None,
    grasp_axis_position_m: float | None = None,
    hand_yaw_rad: float | None = None,
    apply_table_clearance: bool = True,
) -> dict[str, object]:
    """Generate one bounded executable grasp from CAD and the complete pad surfaces."""

    settings = inputs.config.section("nominal_grasp_generation")
    if (
        settings.get("method")
        != "AXIAL_CENTER_OF_MASS_RADIUS_FULL_PAD_TABLE_CLEARANCE_V1"
    ):
        raise ValueError("unsupported nominal grasp generation method")
    ray_count = int(settings["cad_section_ray_count"])
    coarse_step = float(settings["joint_coarse_step_rad"])
    bisections = int(settings["joint_root_bisection_iterations"])
    palm_position = float(
        settings["palm_joint_position_rad"]
        if palm_joint_position_rad is None
        else palm_joint_position_rad
    )
    yaw = float(
        settings.get("hand_yaw_rad", 0.0)
        if hand_yaw_rad is None
        else hand_yaw_rad
    )
    approach_seed_value = settings.get("approach_high_seed_arm_positions_rad")
    approach_seed = (
        None
        if approach_seed_value is None
        else np.asarray(approach_seed_value, dtype=np.float64)
    )
    clearances = np.asarray(
        settings["pregrasp_radial_clearance_m"], dtype=np.float64
    )
    final_increment = float(settings["final_closure_increment_rad"])
    derivative_step = float(settings["closing_derivative_step_rad"])
    minimum_table_clearance = float(
        settings["minimum_pregrasp_hand_table_clearance_m"]
    )
    if (
        ray_count < 3
        or not math.isfinite(coarse_step)
        or coarse_step <= 0.0
        or bisections < 1
        or clearances.shape != (3,)
        or np.any(clearances <= 0.0)
        or not math.isfinite(final_increment)
        or final_increment <= 0.0
        or not math.isfinite(derivative_step)
        or derivative_step <= 0.0
        or not math.isfinite(minimum_table_clearance)
        or minimum_table_clearance < 0.0
        or not math.isfinite(palm_position)
        or not math.isfinite(yaw)
        or not isinstance(apply_table_clearance, bool)
        or (
            approach_seed is not None
            and (approach_seed.shape != (7,) or not np.all(np.isfinite(approach_seed)))
        )
    ):
        raise ValueError("nominal grasp generation bounds are malformed")
    if inputs.task_grip_surfaces is None:
        raise ValueError("complete user-confirmed pad surfaces are unavailable")

    model = inputs.object_contract.model
    basis = np.asarray(
        inputs.object_contract.task_frame_rotation_object, dtype=np.float64
    )
    origin = np.asarray(model.assembly_axis_origin_m, dtype=np.float64)
    vertices_task = (np.asarray(model.mesh.vertices_m) - origin) @ basis
    z_min = float(np.min(vertices_task[:, 2]))
    z_max = float(np.max(vertices_task[:, 2]))
    center_of_mass_task = (np.asarray(model.com_m) - origin) @ basis
    grasp_z = float(
        settings.get("grasp_axis_position_m", center_of_mass_task[2])
        if grasp_axis_position_m is None
        else grasp_axis_position_m
    )
    if not math.isfinite(grasp_z) or not z_min <= grasp_z <= z_max:
        raise ValueError("requested grasp-axis position is outside the CAD")

    import trimesh

    task_mesh = trimesh.Trimesh(
        vertices=vertices_task, faces=model.mesh.faces, process=False
    )
    segments = trimesh.intersections.mesh_plane(
        task_mesh,
        plane_normal=(0.0, 0.0, 1.0),
        plane_origin=(0.0, 0.0, grasp_z),
    )[:, :, :2]
    if not len(segments):
        raise ValueError("CAD reference plane has no closed-section segments")
    segment_start = segments[:, 0]
    segment_vector = segments[:, 1] - segments[:, 0]

    def cross2(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]

    radii = []
    denominator_epsilon = float(settings["ray_parallel_epsilon"])
    segment_tolerance = float(settings["segment_fraction_tolerance"])
    for index in range(ray_count):
        theta = 2.0 * np.pi * index / ray_count
        direction = np.asarray((np.cos(theta), np.sin(theta)))
        denominator = cross2(direction, segment_vector)
        nonparallel = np.abs(denominator) > denominator_epsilon
        distance = np.full(len(segments), np.nan)
        fraction = np.full(len(segments), np.nan)
        distance[nonparallel] = (
            cross2(segment_start[nonparallel], segment_vector[nonparallel])
            / denominator[nonparallel]
        )
        fraction[nonparallel] = (
            cross2(segment_start[nonparallel], direction)
            / denominator[nonparallel]
        )
        hit = (
            nonparallel
            & (distance >= 0.0)
            & (fraction >= -segment_tolerance)
            & (fraction <= 1.0 + segment_tolerance)
        )
        if not np.any(hit):
            raise ValueError(f"CAD section ray {index} has no outward intersection")
        radii.append(float(np.max(distance[hit])))
    object_radius = float(np.median(radii))

    hand = inputs.hand_model
    joint_names = tuple(hand.independent_joint_names)
    expected_names = ("f1j1", "f1j2", "f2j1", "f3j2")
    if joint_names != expected_names:
        raise ValueError("independent hand joint order changed")
    base_positions = np.asarray(
        [hand.joints[name].limit.lower for name in joint_names], dtype=np.float64
    )
    palm_index = joint_names.index("f1j1")
    palm_limit = hand.joints["f1j1"].limit
    if not palm_limit.lower <= palm_position <= palm_limit.upper:
        raise ValueError("shared palm position is outside the hand limit")
    base_positions[palm_index] = palm_position
    fingers = (
        ("finger_1_pad", "f1j2"),
        ("finger_2_pad", "f2j1"),
        ("finger_3_pad", "f3j2"),
    )

    def radial_minimum(
        pad_name: str, joint_name: str, joint_position: float
    ) -> tuple[float, np.ndarray, int]:
        positions = np.array(base_positions, copy=True)
        positions[joint_names.index(joint_name)] = float(joint_position)
        surface = inputs.task_grip_surfaces[pad_name]
        transform = hand.forward_kinematics(positions)[surface.link_name]
        centers = (
            np.mean(surface.triangles_local_m, axis=1) @ transform[:3, :3].T
            + transform[:3, 3]
        )
        normals = surface.face_normals_local @ transform[:3, :3].T
        radius = np.linalg.norm(centers[:, :2], axis=1)
        nonaxis = radius > np.finfo(np.float64).eps
        inward = np.zeros(len(radius), dtype=np.bool_)
        inward[nonaxis] = np.einsum(
            "ij,ij->i",
            normals[nonaxis, :2],
            centers[nonaxis, :2] / radius[nonaxis, None],
        ) < 0.0
        eligible = np.flatnonzero(inward)
        if not len(eligible):
            raise ValueError(f"{pad_name} has no inward-facing full-pad face")
        face = int(eligible[np.argmin(radius[eligible])])
        return float(radius[face]), centers[face], face

    def first_radius_root(
        pad_name: str, joint_name: str, target_radius: float
    ) -> tuple[float, np.ndarray, int]:
        limit = hand.joints[joint_name].limit
        lower, upper = float(limit.lower), float(limit.upper)
        lower_radius = radial_minimum(pad_name, joint_name, lower)[0]
        if lower_radius <= target_radius:
            raise ValueError(f"{pad_name} starts inside its requested radius")
        bracket = None
        previous = lower
        sample = lower + coarse_step
        while previous < upper:
            current = min(sample, upper)
            if radial_minimum(pad_name, joint_name, current)[0] <= target_radius:
                bracket = (previous, current)
                break
            previous = current
            sample += coarse_step
        if bracket is None:
            raise ValueError(f"{pad_name} has no bounded radial first intersection")
        left, right = bracket
        for _ in range(bisections):
            middle = 0.5 * (left + right)
            if radial_minimum(pad_name, joint_name, middle)[0] > target_radius:
                left = middle
            else:
                right = middle
        root = 0.5 * (left + right)
        _, point, face = radial_minimum(pad_name, joint_name, root)
        return root, point, face

    pregrasp = [palm_position]
    contact = [palm_position]
    final = [palm_position]
    contact_points_hand = []
    selected_source_faces = []
    for (pad_name, joint_name), clearance in zip(fingers, clearances):
        pregrasp_root, _, _ = first_radius_root(
            pad_name, joint_name, object_radius + float(clearance)
        )
        contact_root, contact_point, face = first_radius_root(
            pad_name, joint_name, object_radius
        )
        limit = hand.joints[joint_name].limit
        final_root = min(contact_root + final_increment, float(limit.upper))
        if not pregrasp_root < contact_root < final_root:
            raise ValueError(f"{pad_name} radial roots are not ordered")
        next_position = min(contact_root + derivative_step, float(limit.upper))
        if radial_minimum(pad_name, joint_name, next_position)[0] >= object_radius:
            raise ValueError(f"{pad_name} does not move inward after first contact")
        pregrasp.append(pregrasp_root)
        contact.append(contact_root)
        final.append(final_root)
        contact_points_hand.append(contact_point)
        selected_source_faces.append(int(
            inputs.task_grip_surfaces[pad_name].source_face_indices[face]
        ))

    contact_points_hand = np.asarray(contact_points_hand, dtype=np.float64)
    hand_origin_task = np.asarray(
        (0.0, 0.0, grasp_z - float(np.mean(contact_points_hand[:, 2])))
    )
    if yaw == 0.0:
        yaw_rotation = np.eye(3, dtype=np.float64)
        grasp_rotation = basis
        contact_points_task = contact_points_hand + hand_origin_task
    else:
        cosine, sine = math.cos(yaw), math.sin(yaw)
        yaw_rotation = np.asarray(
            ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        grasp_rotation = basis @ yaw_rotation
        contact_points_task = (
            contact_points_hand @ yaw_rotation.T + hand_origin_task
        )
    angles = np.sort(np.mod(
        np.arctan2(contact_points_task[:, 1], contact_points_task[:, 0]),
        2.0 * np.pi,
    ))
    angular_gaps = np.diff(np.r_[angles, angles[0] + 2.0 * np.pi])
    if float(np.max(angular_gaps)) >= np.pi:
        raise ValueError("predicted contacts do not surround the object axis")

    object_from_hand = np.eye(4, dtype=np.float64)
    object_from_hand[:3, :3] = grasp_rotation
    object_from_hand[:3, 3] = origin + basis @ hand_origin_task
    contact_points_object = contact_points_task @ basis.T + origin

    pregrasp_transforms = hand.forward_kinematics(np.asarray(pregrasp))
    world_from_hand = inputs.frozen_world_from_object @ object_from_hand
    minimum_world_z = math.inf
    limiting_link = ""
    for link_name, triangles in inputs.hand_collision_triangles_by_link.items():
        link_from_local = pregrasp_transforms[link_name]
        points_local = np.asarray(triangles, dtype=np.float64).reshape(-1, 3)
        points_hand = (
            points_local @ link_from_local[:3, :3].T
            + link_from_local[:3, 3]
        )
        points_world = (
            points_hand @ world_from_hand[:3, :3].T
            + world_from_hand[:3, 3]
        )
        link_minimum_world_z = float(np.min(points_world[:, 2]))
        if link_minimum_world_z < minimum_world_z:
            minimum_world_z = link_minimum_world_z
            limiting_link = link_name
    initial_table_clearance = minimum_world_z - float(inputs.table_top_z_m)
    table_clearance_shift = (
        max(0.0, minimum_table_clearance - initial_table_clearance)
        if apply_table_clearance
        else 0.0
    )
    if table_clearance_shift > 0.0:
        shift_world = np.asarray((0.0, 0.0, table_clearance_shift))
        shift_object = (
            inputs.frozen_world_from_object[:3, :3].T @ shift_world
        )
        object_from_hand[:3, 3] += shift_object
        contact_points_object += shift_object

    adjusted_contact_points_task = (contact_points_object - origin) @ basis
    if np.any(adjusted_contact_points_task[:, 2] < z_min) or np.any(
        adjusted_contact_points_task[:, 2] > z_max
    ):
        raise ValueError("table-cleared full-pad contact lies outside CAD axial bounds")
    triangle_area = 0.5 * float(np.linalg.norm(np.cross(
        contact_points_object[1] - contact_points_object[0],
        contact_points_object[2] - contact_points_object[0],
    )))
    control_plan = {
        "object_from_hand_row_major": object_from_hand.ravel().tolist(),
        "approach_direction_object": model.assembly_axis.tolist(),
        "pregrasp_joint_positions_rad": list(map(float, pregrasp)),
        "final_joint_positions_rad": list(map(float, final)),
    }
    if approach_seed is not None:
        control_plan["approach_high_seed_arm_positions_rad"] = approach_seed.tolist()
    return {
        "method": settings["method"],
        "control_plan": control_plan,
        "evidence": {
            "derived_only_from_current_object_cad_shared_hand_and_frozen_table_geometry": True,
            "cad_reference_section_z_m": grasp_z,
            "palm_joint_position_rad": palm_position,
            "hand_yaw_rad": yaw,
            "cad_section_outer_radius_m": object_radius,
            "cad_section_ray_radius_range_m": [min(radii), max(radii)],
            "predicted_contact_joint_positions_rad": list(map(float, contact)),
            "predicted_contact_points_object_m": contact_points_object.tolist(),
            "pregrasp_limiting_hand_link": limiting_link,
            "pregrasp_initial_hand_table_clearance_m": initial_table_clearance,
            "minimum_pregrasp_hand_table_clearance_m": minimum_table_clearance,
            "applied_hand_table_clearance_shift_world_z_m": table_clearance_shift,
            "predicted_contact_source_faces_diagnostic_only": selected_source_faces,
            "predicted_contact_maximum_angular_gap_rad": float(
                np.max(angular_gaps)
            ),
            "predicted_contact_triangle_area_m2": triangle_area,
        },
    }


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
    "generate_axial_pad_intersection_grasp",
    "load_task_grip_surfaces",
    "motion_faces_current_workspace",
    "motion_compatible_with_object_witness",
    "task_noncontact_triangles",
]
