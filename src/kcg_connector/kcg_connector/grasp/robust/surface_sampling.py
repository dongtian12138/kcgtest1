"""Semantic surface extraction and deterministic area sampling for CARTS-Grasp.

Direction feasibility is supplied by the hand contact model as a convex cone.
This object-side module contains no connector-specific normal thresholds or
candidate coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Callable, Iterable, Sequence

import numpy as np

from kcg_connector.grasp.robust.object_model import ObjectGraspModel
from kcg_connector.grasp.robust.triangle_canonicalization import (
    RegisteredTaskFrame,
    UNORIENTED_CANONICAL_REPRESENTATIVE_NORMAL_DIAGNOSTIC_ONLY,
    canonical_representative_normals,
    canonicalize_unoriented_triangles,
)


AREA_STRATIFIED_SOBOL = "AREA_STRATIFIED_SOBOL"
CYLINDRICAL_NORMAL_FRAME = "OUTWARD_RADIAL_TANGENTIAL_ASSEMBLY_AXIS"

VisibilityPredicate = Callable[[ObjectGraspModel], Sequence[bool]]


def _readonly_array(
    value: object, dtype: object, shape_tail: tuple[int, ...], name: str
) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    tail_start = array.ndim - len(shape_tail)
    if array.ndim < len(shape_tail) or tuple(array.shape[tail_start:]) != shape_tail:
        raise ValueError(f"{name} must end in shape {shape_tail}, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    array.setflags(write=False)
    return array


def _floating_tolerance(*arrays: np.ndarray, multiplier: float = 256.0) -> float:
    scale = 1.0
    for array in arrays:
        if array.size:
            scale = max(scale, float(np.max(np.abs(array))))
    return multiplier * np.finfo(np.float64).eps * scale


@dataclass(frozen=True)
class PadNormalCone:
    """Hand-derived polyhedral cone for feasible object surface normals.

    Rows of ``halfspaces_local`` encode ``A @ n_local <= 0``.  The local
    coordinates are ``[outward radial, tangential, assembly axis]``.  A hand
    model can derive these homogeneous halfspaces from PAD orientation range,
    kinematics, and its contact footprint without placing object-specific
    angular thresholds in this module.
    """

    halfspaces_local: np.ndarray
    source: str
    frame_convention: str = CYLINDRICAL_NORMAL_FRAME

    def __post_init__(self) -> None:
        halfspaces = _readonly_array(
            self.halfspaces_local,
            dtype=np.float64,
            shape_tail=(3,),
            name="halfspaces_local",
        )
        if halfspaces.ndim != 2 or len(halfspaces) == 0:
            raise ValueError("PadNormalCone needs at least one three-dimensional halfspace")
        row_norms = np.linalg.norm(halfspaces, axis=1)
        if np.any(row_norms == 0.0):
            raise ValueError("PadNormalCone halfspace rows must be non-zero")
        normalized = np.array(halfspaces / row_norms[:, None], copy=True)
        normalized.setflags(write=False)
        if not str(self.source):
            raise ValueError("PadNormalCone source must identify its HandContactModel")
        if self.frame_convention != CYLINDRICAL_NORMAL_FRAME:
            raise ValueError("unsupported PadNormalCone frame convention")
        object.__setattr__(self, "halfspaces_local", normalized)

    def contains(self, normals_local: Sequence[Sequence[float]]) -> np.ndarray:
        normals = np.asarray(normals_local, dtype=np.float64)
        if normals.ndim != 2 or normals.shape[1] != 3 or not np.all(np.isfinite(normals)):
            raise ValueError("normals_local must be a finite array with shape (N, 3)")
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths == 0.0):
            raise ValueError("normal vectors cannot be zero")
        unit_normals = normals / lengths[:, None]
        residuals = unit_normals @ self.halfspaces_local.T
        tolerance = _floating_tolerance(unit_normals, self.halfspaces_local)
        return np.all(residuals <= tolerance, axis=1)

    def intersects_normal_lines(
        self, normals_local: Sequence[Sequence[float]]
    ) -> np.ndarray:
        """Return whether either orientation of each normal line is in the cone.

        The two directed cone queries are deliberately kept separate.  Taking
        absolute half-space residuals would define a different, generally
        smaller set for an asymmetric cone.
        """

        normals = np.asarray(normals_local, dtype=np.float64)
        return self.contains(normals) | self.contains(-normals)

    @property
    def contract_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CARTS_GRASP_PAD_NORMAL_CONE_V1\0")
        digest.update(np.asarray(self.halfspaces_local, dtype="<f8").tobytes(order="C"))
        digest.update(self.source.encode("utf-8"))
        digest.update(self.frame_convention.encode("utf-8"))
        return digest.hexdigest()


@dataclass(frozen=True)
class SurfacePatch:
    """A connected, semantics-homogeneous set of feasible mesh faces."""

    patch_id: str
    face_indices: np.ndarray
    semantic: str
    area_m2: float
    area_centroid_m: np.ndarray
    area_normal_second_moment_m2: np.ndarray

    def __post_init__(self) -> None:
        indices = np.array(self.face_indices, dtype=np.int64, copy=True)
        centroid = np.array(self.area_centroid_m, dtype=np.float64, copy=True)
        second_moment = np.array(
            self.area_normal_second_moment_m2,
            dtype=np.float64,
            copy=True,
        )
        if indices.ndim != 1 or len(indices) == 0 or np.any(indices < 0):
            raise ValueError("SurfacePatch face_indices must be a non-empty non-negative vector")
        if centroid.shape != (3,) or second_moment.shape != (3, 3):
            raise ValueError(
                "SurfacePatch centroid must be a three-vector and normal "
                "second moment must be 3x3"
            )
        if not np.all(np.isfinite(centroid)) or not np.all(np.isfinite(second_moment)):
            raise ValueError("SurfacePatch vectors must be finite")
        if not np.isfinite(self.area_m2) or float(self.area_m2) <= 0.0:
            raise ValueError("SurfacePatch area_m2 must be finite and positive")
        if not str(self.patch_id) or not str(self.semantic):
            raise ValueError("SurfacePatch identity and semantic cannot be empty")
        tolerance = _floating_tolerance(
            second_moment,
            np.asarray((self.area_m2,), dtype=np.float64),
        )
        symmetric_second_moment = 0.5 * (second_moment + second_moment.T)
        if not np.allclose(
            second_moment,
            second_moment.T,
            rtol=0.0,
            atol=tolerance,
        ) or float(np.min(np.linalg.eigvalsh(symmetric_second_moment))) < -tolerance:
            raise ValueError(
                "SurfacePatch normal second moment must be symmetric positive semidefinite"
            )
        if not np.isclose(
            np.trace(symmetric_second_moment),
            float(self.area_m2),
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError(
                "SurfacePatch normal second-moment trace must equal patch area"
            )
        indices.setflags(write=False)
        centroid.setflags(write=False)
        symmetric_second_moment.setflags(write=False)
        object.__setattr__(self, "face_indices", indices)
        object.__setattr__(self, "area_centroid_m", centroid)
        object.__setattr__(
            self,
            "area_normal_second_moment_m2",
            symmetric_second_moment,
        )
        object.__setattr__(self, "area_m2", float(self.area_m2))


@dataclass(frozen=True)
class SurfaceSamples:
    """Equal-area quasi-Monte-Carlo samples on feasible surface patches.

    ``normals`` are deterministic representatives of unoriented normal lines
    induced by canonical vertex order.  They are diagnostic only and do not
    assert outward/inward orientation or define a contact force-cone axis.
    """

    positions_m: np.ndarray
    normals: np.ndarray
    barycentric_coordinates: np.ndarray
    face_indices: np.ndarray
    patch_ids: tuple[str, ...]
    semantics: tuple[str, ...]
    integration_weights_m2: np.ndarray
    seed: int
    method: str = AREA_STRATIFIED_SOBOL
    normal_semantics: str = (
        UNORIENTED_CANONICAL_REPRESENTATIVE_NORMAL_DIAGNOSTIC_ONLY
    )

    def __post_init__(self) -> None:
        positions = _readonly_array(self.positions_m, np.float64, (3,), "positions_m")
        normals = _readonly_array(self.normals, np.float64, (3,), "normals")
        barycentric = _readonly_array(
            self.barycentric_coordinates,
            np.float64,
            (3,),
            "barycentric_coordinates",
        )
        face_indices = np.array(self.face_indices, dtype=np.int64, copy=True)
        weights = np.array(self.integration_weights_m2, dtype=np.float64, copy=True)
        sample_count = len(positions)
        if (
            positions.ndim != 2
            or normals.shape != positions.shape
            or barycentric.shape != positions.shape
        ):
            raise ValueError("sample positions, normals, and barycentric coordinates must be Nx3")
        if face_indices.shape != (sample_count,) or weights.shape != (sample_count,):
            raise ValueError("sample face indices and integration weights must be length N")
        if len(self.patch_ids) != sample_count or len(self.semantics) != sample_count:
            raise ValueError("sample patch IDs and semantics must be length N")
        if sample_count == 0 or np.any(face_indices < 0):
            raise ValueError("SurfaceSamples cannot be empty")
        if not np.allclose(
            np.sum(barycentric, axis=1),
            np.ones(sample_count),
            rtol=0.0,
            atol=_floating_tolerance(barycentric),
        ) or np.any(barycentric < 0.0):
            raise ValueError("invalid triangle barycentric coordinates")
        if not np.allclose(
            np.linalg.norm(normals, axis=1),
            np.ones(sample_count),
            rtol=0.0,
            atol=_floating_tolerance(normals),
        ):
            raise ValueError("sample normals must be unit vectors")
        if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("integration weights must be finite and positive")
        face_indices.setflags(write=False)
        weights.setflags(write=False)
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "barycentric_coordinates", barycentric)
        object.__setattr__(self, "face_indices", face_indices)
        object.__setattr__(self, "integration_weights_m2", weights)
        object.__setattr__(self, "patch_ids", tuple(self.patch_ids))
        object.__setattr__(self, "semantics", tuple(self.semantics))
        object.__setattr__(self, "seed", int(self.seed))
        if self.method != AREA_STRATIFIED_SOBOL:
            raise ValueError("unsupported surface sampling method")
        if self.normal_semantics != (
            UNORIENTED_CANONICAL_REPRESENTATIVE_NORMAL_DIAGNOSTIC_ONLY
        ):
            raise ValueError("unsupported SurfaceSamples normal semantics")


def cylindrical_normal_coordinates(
    model: ObjectGraspModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source representatives of normal lines in cylindrical coordinates.

    The second return value marks centroids that have a defined radial
    direction relative to the assembly-axis line.
    """

    centroids = model.mesh.face_centroids_m
    normals = model.mesh.face_normals
    delta = centroids - model.assembly_axis_origin_m
    axial_coordinate = delta @ model.assembly_axis
    radial = delta - axial_coordinate[:, None] * model.assembly_axis
    radial_lengths = np.linalg.norm(radial, axis=1)
    valid = radial_lengths > 0.0
    radial_unit = np.zeros_like(radial)
    radial_unit[valid] = radial[valid] / radial_lengths[valid, None]
    tangential_unit = np.cross(
        np.broadcast_to(model.assembly_axis, radial_unit.shape), radial_unit
    )
    local = np.column_stack(
        (
            np.einsum("ij,ij->i", normals, radial_unit),
            np.einsum("ij,ij->i", normals, tangential_unit),
            normals @ model.assembly_axis,
        )
    )
    return local, valid


def eligible_lateral_face_mask(
    model: ObjectGraspModel,
    pad_normal_cone: PadNormalCone,
    *,
    require_watertight: bool = True,
    allowed_face_mask: Sequence[bool] | None = None,
    visibility_predicate: VisibilityPredicate | None = None,
) -> np.ndarray:
    """Select semantic faces whose unoriented normal lines intersect the PAD cone.

    A caller may additionally provide a contract ``allowed_face_mask`` and/or
    a geometry-only visibility predicate (for example, first intersection of
    an external closure ray).  With neither argument, this function does not
    claim that generic ``external_surface`` labels identify or exclude mating,
    pin, thread, or cavity surfaces.
    """

    if require_watertight and not model.mesh.is_watertight:
        raise ValueError("external lateral extraction requires a watertight object mesh")
    local_normals, radial_valid = cylindrical_normal_coordinates(model)
    cone_valid = np.zeros(len(model.mesh.faces), dtype=bool)
    if np.any(radial_valid):
        cone_valid[radial_valid] = pad_normal_cone.intersects_normal_lines(
            local_normals[radial_valid]
        )
    accessibility = np.ones(len(model.mesh.faces), dtype=bool)
    if allowed_face_mask is not None:
        contract_mask = np.asarray(allowed_face_mask, dtype=bool)
        if contract_mask.shape != (len(model.mesh.faces),):
            raise ValueError("allowed_face_mask must contain one Boolean per mesh face")
        accessibility &= contract_mask
    if visibility_predicate is not None:
        visibility_mask = np.asarray(visibility_predicate(model), dtype=bool)
        if visibility_mask.shape != (len(model.mesh.faces),):
            raise ValueError("visibility_predicate must return one Boolean per mesh face")
        accessibility &= visibility_mask
    mask = model.contact_face_mask & radial_valid & cone_valid & accessibility
    mask.setflags(write=False)
    return mask


def _patch_identifier(
    model: ObjectGraspModel, semantic: str, face_indices: Sequence[int]
) -> str:
    digest = hashlib.sha256()
    digest.update(b"CARTS_GRASP_SURFACE_PATCH_V1\0")
    digest.update(model.provenance.source_sha256.encode("ascii"))
    digest.update(semantic.encode("utf-8"))
    indices = np.asarray(tuple(face_indices), dtype="<i8")
    digest.update(indices.tobytes(order="C"))
    return f"patch-{digest.hexdigest()[:20]}"


def extract_lateral_contact_patches(
    model: ObjectGraspModel,
    pad_normal_cone: PadNormalCone,
    *,
    require_watertight: bool = True,
    allowed_face_mask: Sequence[bool] | None = None,
    visibility_predicate: VisibilityPredicate | None = None,
) -> tuple[SurfacePatch, ...]:
    """Extract connected feasible patches without connector-specific geometry."""

    eligible = eligible_lateral_face_mask(
        model,
        pad_normal_cone,
        require_watertight=require_watertight,
        allowed_face_mask=allowed_face_mask,
        visibility_predicate=visibility_predicate,
    )
    eligible_indices = set(int(value) for value in np.flatnonzero(eligible))
    if not eligible_indices:
        return ()

    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_index in sorted(eligible_indices):
        face = model.mesh.faces[face_index]
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (min(int(first), int(second)), max(int(first), int(second)))
            edge_faces.setdefault(edge, []).append(face_index)
    neighbors: dict[int, set[int]] = {index: set() for index in eligible_indices}
    for entries in edge_faces.values():
        for face_index in entries:
            for other in entries:
                if (
                    other != face_index
                    and model.mesh.face_semantics[other]
                    == model.mesh.face_semantics[face_index]
                ):
                    neighbors[face_index].add(other)

    areas = model.mesh.face_areas_m2
    centroids = model.mesh.face_centroids_m
    patches: list[SurfacePatch] = []
    unseen = set(eligible_indices)
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        stack = [root]
        component: list[int] = []
        while stack:
            face_index = stack.pop()
            component.append(face_index)
            for neighbor in sorted(neighbors[face_index], reverse=True):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        component.sort()
        indices = np.asarray(component, dtype=np.int64)
        area = float(np.sum(areas[indices]))
        centroid = np.sum(areas[indices, None] * centroids[indices], axis=0) / area
        normal_representatives = model.mesh.face_normals[indices]
        normal_second_moment = np.einsum(
            "i,ij,ik->jk",
            areas[indices],
            normal_representatives,
            normal_representatives,
        )
        semantic = model.mesh.face_semantics[component[0]]
        patches.append(
            SurfacePatch(
                patch_id=_patch_identifier(model, semantic, component),
                face_indices=indices,
                semantic=semantic,
                area_m2=area,
                area_centroid_m=centroid,
                area_normal_second_moment_m2=normal_second_moment,
            )
        )
    return tuple(patches)


def _sobol_points(sample_count: int, seed: int) -> np.ndarray:
    if int(sample_count) != sample_count or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    try:
        from scipy.stats import qmc
    except ImportError as error:
        raise RuntimeError(
            "AREA_STRATIFIED_SOBOL requires scipy.stats.qmc; no random fallback is allowed"
        ) from error
    power = int(math.ceil(math.log2(sample_count))) if sample_count > 1 else 0
    sampler = qmc.Sobol(d=3, scramble=True, seed=int(seed))
    points = sampler.random_base2(m=power)
    return np.asarray(points[:sample_count], dtype=np.float64)


def sample_mesh_faces_area_stratified(
    model: ObjectGraspModel,
    *,
    task_frame: RegisteredTaskFrame,
    face_indices: Iterable[int],
    sample_count: int,
    seed: int,
    face_patch_ids: dict[int, str] | None = None,
) -> SurfaceSamples:
    """Sample specified faces by their SI area using a fixed Sobol sequence."""

    indices = np.asarray(tuple(int(value) for value in face_indices), dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("face_indices must be non-empty")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("face_indices must not contain duplicates")
    if np.any(indices < 0) or np.any(indices >= len(model.mesh.faces)):
        raise ValueError("face_indices contains an out-of-range index")
    indices = np.sort(indices)
    areas = model.mesh.face_areas_m2[indices]
    total_area = float(np.sum(areas))
    if not np.isfinite(total_area) or total_area <= 0.0:
        raise ValueError("selected faces must have positive finite total area")

    unit_samples = _sobol_points(int(sample_count), int(seed))
    cumulative_area = np.cumsum(areas) / total_area
    local_face = np.searchsorted(cumulative_area, unit_samples[:, 0], side="right")
    local_face = np.minimum(local_face, len(indices) - 1)
    selected_faces = indices[local_face]

    root = np.sqrt(unit_samples[:, 1])
    barycentric = np.column_stack(
        (1.0 - root, root * (1.0 - unit_samples[:, 2]), root * unit_samples[:, 2])
    )
    triangles = canonicalize_unoriented_triangles(
        model.mesh.face_vertices_m[selected_faces],
        task_frame=task_frame,
    )
    positions = np.einsum("ni,nij->nj", barycentric, triangles)
    normals = canonical_representative_normals(triangles)
    weights = np.full(int(sample_count), total_area / int(sample_count), dtype=np.float64)
    patch_lookup = {} if face_patch_ids is None else dict(face_patch_ids)
    patch_ids = tuple(patch_lookup.get(int(index), "unassigned") for index in selected_faces)
    semantics = tuple(model.mesh.face_semantics[int(index)] for index in selected_faces)
    return SurfaceSamples(
        positions_m=positions,
        normals=normals,
        barycentric_coordinates=barycentric,
        face_indices=selected_faces,
        patch_ids=patch_ids,
        semantics=semantics,
        integration_weights_m2=weights,
        seed=int(seed),
    )


def sample_contact_surfaces(
    model: ObjectGraspModel,
    pad_normal_cone: PadNormalCone,
    *,
    task_frame: RegisteredTaskFrame,
    sample_count: int,
    seed: int,
    require_watertight: bool = True,
    allowed_face_mask: Sequence[bool] | None = None,
    visibility_predicate: VisibilityPredicate | None = None,
) -> SurfaceSamples:
    """Extract feasible patches and sample their union with no model-specific tuning."""

    patches = extract_lateral_contact_patches(
        model,
        pad_normal_cone,
        require_watertight=require_watertight,
        allowed_face_mask=allowed_face_mask,
        visibility_predicate=visibility_predicate,
    )
    if not patches:
        raise ValueError("object and PAD normal cone have no feasible contact surface")
    face_patch_ids = {
        int(face_index): patch.patch_id
        for patch in patches
        for face_index in patch.face_indices
    }
    return sample_mesh_faces_area_stratified(
        model,
        task_frame=task_frame,
        face_indices=face_patch_ids,
        sample_count=sample_count,
        seed=seed,
        face_patch_ids=face_patch_ids,
    )


__all__ = [
    "AREA_STRATIFIED_SOBOL",
    "CYLINDRICAL_NORMAL_FRAME",
    "PadNormalCone",
    "RegisteredTaskFrame",
    "SurfacePatch",
    "SurfaceSamples",
    "VisibilityPredicate",
    "UNORIENTED_CANONICAL_REPRESENTATIVE_NORMAL_DIAGNOSTIC_ONLY",
    "cylindrical_normal_coordinates",
    "eligible_lateral_face_mask",
    "extract_lateral_contact_patches",
    "sample_contact_surfaces",
    "sample_mesh_faces_area_stratified",
]
