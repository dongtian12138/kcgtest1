"""Hash-bound outward normals and mature-mesh PAD/object proximity queries."""
from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np
from scipy.spatial import cKDTree

try:
    import fcl
except ImportError:
    fcl = None
try:
    import trimesh
except (ImportError, TypeError):
    trimesh = None
from kcg_connector.grasp.carts_v2.fast_filter import build_fcl_bvh_model
from kcg_connector.grasp.carts_v2.models import (
    HARD_FORBIDDEN, PRIMARY_GRIP, SECONDARY_GRIP,
)
@dataclass(frozen=True)
class NearestSurface:
    point_m: np.ndarray
    distance_m: np.ndarray
    face_index: np.ndarray
    intersecting: bool = False
    surface_face_index: np.ndarray | None = None
    surface_normal_m: np.ndarray | None = None
    surface_legacy_blue_pad: np.ndarray | None = None
    surface_patch_index: np.ndarray | None = None
    surface_patch_area_m2: np.ndarray | None = None
    object_role_code: np.ndarray | None = None
    forbidden_first_contact: bool = False
    forbidden_face_index: int | None = None
    forbidden_distance_m: float | None = None
    registered_patch_count: int | None = None
    finite_patch_witness_count: int | None = None
    contact_region_pass: bool = False
    contact_region_witness_count: int = 0
    contact_region_triangle_area_m2: float = 0.0
    region_primary_sampled_hand_patch_area_fraction: float | None = None
    region_secondary_sampled_hand_patch_area_fraction: float | None = None
    region_composite_normal_m: np.ndarray | None = None
    region_normal_dispersion_rad: float | None = None


def material_bound_object_face_normals(inputs) -> np.ndarray:
    """Return source-face normals bound to the registered material orientation."""

    loaded = inputs.object_contract
    mesh = loaded.model.mesh
    certificate = loaded.orientation_certificate
    material = loaded.material_boundary_evidence
    signs = np.asarray(
        certificate.positive_volume_winding_sign_by_source_face,
        dtype=np.float64,
    )
    if (
        material.formal_material_boundary_eligible is not True
        or material.certificate.orientation_certificate_sha256
        != certificate.canonical_sha256
        or signs.shape != (len(mesh.faces),)
        or not np.all(np.isin(signs, (-1.0, 1.0)))
    ):
        raise ValueError("object outward-normal material binding is incomplete")
    normals = np.asarray(mesh.face_normals, dtype=np.float64) * signs[:, None]
    normals.setflags(write=False)
    return normals

def _closest_points_on_triangles(triangles: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Vectorized closest points for one point paired with one triangle."""

    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac, ap = b - a, c - a, points - a
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    result, assigned = np.empty_like(points), (d1 <= 0.0) & (d2 <= 0.0)
    result[assigned] = a[assigned]
    bp = points - b
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    mask = (~assigned) & (d3 >= 0.0) & (d4 <= d3)
    result[mask], assigned = b[mask], assigned | mask
    vc = d1 * d4 - d3 * d2
    mask = (~assigned) & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    fraction = np.divide(d1, d1 - d3, out=np.zeros_like(d1), where=d1 != d3)
    result[mask], assigned = a[mask] + fraction[mask, None] * ab[mask], assigned | mask
    cp = points - c
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    mask = (~assigned) & (d6 >= 0.0) & (d5 <= d6)
    result[mask], assigned = c[mask], assigned | mask
    vb = d5 * d2 - d1 * d6
    mask = (~assigned) & (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    fraction = np.divide(d2, d2 - d6, out=np.zeros_like(d2), where=d2 != d6)
    result[mask], assigned = a[mask] + fraction[mask, None] * ac[mask], assigned | mask
    va, edge = d3 * d6 - d5 * d4, (d4 - d3) + (d5 - d6)
    mask = (~assigned) & (va <= 0.0) & (d4 >= d3) & (d5 >= d6)
    fraction = np.divide(d4 - d3, edge, out=np.zeros_like(edge), where=edge != 0.0)
    result[mask], assigned = b[mask] + fraction[mask, None] * (c - b)[mask], assigned | mask
    inverse = np.divide(1.0, va + vb + vc, out=np.zeros_like(va), where=(va + vb + vc) != 0)
    v, w = vb * inverse, vc * inverse
    result[~assigned] = (a[~assigned] + v[~assigned, None] * ab[~assigned]
                         + w[~assigned, None] * ac[~assigned])
    return result


class LegacyMeshProximityIndex:
    """Centroid-neighborhood baseline retained outside the production exact path."""

    def __init__(self, inputs) -> None:
        self._triangles = np.asarray(inputs.object_contract.model.mesh.face_vertices_m)
        self._tree = cKDTree(np.mean(self._triangles, axis=1))
        self._neighbor_count = int(inputs.config.section(
            "closure_prediction")["nearest_face_candidate_count"])

    def query(self, points_m: np.ndarray) -> NearestSurface:
        count = min(self._neighbor_count, len(self._triangles))
        _distance, face_ids = self._tree.query(points_m, k=count)
        face_ids = np.asarray(face_ids, dtype=np.int64)
        if count == 1:
            face_ids = face_ids[:, None]
        repeated = np.repeat(points_m, count, axis=0)
        closest = _closest_points_on_triangles(
            self._triangles[face_ids.reshape(-1)], repeated).reshape(len(points_m), count, 3)
        distances = np.linalg.norm(closest - points_m[:, None, :], axis=2)
        selected, rows = np.argmin(distances, axis=1), np.arange(len(points_m))
        return NearestSurface(closest[rows, selected], distances[rows, selected],
                              face_ids[rows, selected])


def nearest_motion_compatible_index(
    distance_m: np.ndarray,
    inward_motion_m_per_phase: np.ndarray,
    minimum_inward_motion_m_per_phase: float,
) -> int:
    """Select the nearest compatible witness, or explicitly return an empty set."""

    distance = np.asarray(distance_m, dtype=np.float64)
    inward = np.asarray(inward_motion_m_per_phase, dtype=np.float64)
    if distance.shape != inward.shape or distance.ndim != 1:
        raise ValueError("distance and inward motion must be equal one-dimensional arrays")
    eligible = (
        np.isfinite(distance)
        & np.isfinite(inward)
        & (inward >= float(minimum_inward_motion_m_per_phase))
    )
    if not np.any(eligible):
        return -1
    return int(np.argmin(np.where(eligible, distance, np.inf)))


def _contact_region(nearest, normals, compatible, distance_limit, minimum_area):
    patches, areas, roles = (nearest.surface_patch_index,
                             nearest.surface_patch_area_m2,
                             nearest.object_role_code)
    if patches is None or areas is None or roles is None:
        return {}
    eligible = (np.asarray(compatible, dtype=np.bool_)
                & np.isfinite(nearest.distance_m)
                & (nearest.distance_m <= float(distance_limit)
                   + 64.0 * np.finfo(np.float64).eps))
    indices = np.flatnonzero(eligible)
    if len(indices) < 3 or len(np.unique(patches[indices])) < 3:
        return {"contact_region_witness_count": int(len(indices))}
    points = nearest.point_m[indices]
    triangle_area = max(
        (0.5 * float(np.linalg.norm(np.cross(points[j] - points[i],
                                             points[k] - points[i])))
         for i in range(len(points) - 2) for j in range(i + 1, len(points) - 1)
         for k in range(j + 1, len(points))), default=0.0)
    weights = np.asarray(areas[indices], dtype=np.float64)
    vector = np.sum(normals[indices] * weights[:, None], axis=0)
    norm = float(np.linalg.norm(vector))
    if triangle_area < float(minimum_area) or norm <= np.finfo(np.float64).eps:
        return {"contact_region_witness_count": int(len(indices)),
                "contact_region_triangle_area_m2": triangle_area}
    composite = vector / norm
    angles = np.arccos(np.clip(normals[indices] @ composite, -1.0, 1.0))
    total = float(np.sum(weights))
    return {"contact_region_pass": True,
            "contact_region_witness_count": int(len(indices)),
            "contact_region_triangle_area_m2": triangle_area,
            "region_primary_sampled_hand_patch_area_fraction": float(np.sum(
                weights[roles[indices] == PRIMARY_GRIP]) / total),
            "region_secondary_sampled_hand_patch_area_fraction": float(np.sum(
                weights[roles[indices] == SECONDARY_GRIP]) / total),
            "region_composite_normal_m": composite,
            "region_normal_dispersion_rad": float(np.sqrt(
                np.sum(weights * angles * angles) / total))}


class ExactContactSurfaceQuery:
    """FCL witness between the full object and one registered hand surface."""

    def __init__(self, inputs) -> None:
        if fcl is None:
            raise RuntimeError("FCL_EXACT_PAD_MESH requires python-fcl")
        loaded = inputs.object_contract
        mesh = loaded.model.mesh
        self._normals = material_bound_object_face_normals(inputs)
        self._face_count = len(mesh.faces)
        self._object = fcl.CollisionObject(
            build_fcl_bvh_model(mesh.vertices_m, mesh.faces)
        )
        task_surfaces = getattr(inputs, "task_grip_surfaces", None)
        allowed = np.ones(self._face_count, dtype=np.bool_)
        self._face_roles = np.full(self._face_count, PRIMARY_GRIP, dtype=np.uint8)
        roles = getattr(inputs, "face_roles", None)
        if roles is not None:
            allowed = np.asarray(roles.face_is_allowed, dtype=np.bool_)
            if allowed.shape != (self._face_count,) or not np.any(allowed):
                raise ValueError("allowed object contact surface is empty or malformed")
            self._face_roles = np.asarray(getattr(
                roles, "face_role",
                np.where(allowed, PRIMARY_GRIP, HARD_FORBIDDEN)), dtype=np.uint8)
            if self._face_roles.shape != (self._face_count,):
                raise ValueError("object surface role domain is malformed")
        self._allowed_object_faces = np.flatnonzero(allowed)
        self._allowed_object = fcl.CollisionObject(build_fcl_bvh_model(
            mesh.vertices_m, mesh.faces[self._allowed_object_faces]))
        self._forbidden_object_faces = np.flatnonzero(~allowed)
        self._forbidden_object = (
            None if len(self._forbidden_object_faces) == 0 else
            fcl.CollisionObject(build_fcl_bvh_model(
                mesh.vertices_m, mesh.faces[self._forbidden_object_faces]))
        )
        geometry = (
            {
                pad.name: (
                    pad.points_local_m,
                    pad.faces,
                    np.zeros((len(pad.faces), 3), dtype=np.float64),
                    np.arange(len(pad.faces), dtype=np.int64),
                    np.ones(len(pad.faces), dtype=np.bool_),
                    np.zeros(len(pad.faces), dtype=np.int64),
                )
                for pad in inputs.hand_contract.pads
            }
            if task_surfaces is None
            else {
                pad_name: (
                    surface.points_local_m,
                    surface.faces,
                    surface.face_normals_local,
                    surface.source_face_indices,
                    surface.legacy_blue_pad_face_mask,
                    surface.patch_indices,
                )
                for pad_name, surface in task_surfaces.items()
            }
        )
        self._task_surface_mode = task_surfaces is not None
        self._pads = {
            name: fcl.CollisionObject(build_fcl_bvh_model(points, faces))
            for name, (points, faces, _normals, _source, _legacy, _patch) in geometry.items()
        }
        self._surface_normals_local = {
            name: np.asarray(row[2]) for name, row in geometry.items()
        }
        self._surface_face_count = {
            name: len(row[1]) for name, row in geometry.items()
        }
        self._surface_source_faces = {name: np.asarray(row[3])
                                      for name, row in geometry.items()}
        self._surface_legacy_blue = {name: np.asarray(row[4])
                                     for name, row in geometry.items()}
        self._patches = {
            name: self._build_patch_objects(*row[:2], np.asarray(row[5]))
            for name, row in geometry.items()
        }
        self._minimum_region_area_m2 = (None if not self._task_surface_mode else
            float(inputs.config.section(
                "fast_filter")["minimum_three_contact_triangle_area_m2"]))
        self._proximity = None
        if trimesh is not None:
            object_mesh = trimesh.Trimesh(
                vertices=mesh.vertices_m, faces=mesh.faces, process=False
            )
            self._proximity = trimesh.proximity.ProximityQuery(object_mesh)
        self._distance_request = fcl.DistanceRequest(enable_nearest_points=True)
        self._collision_request = fcl.CollisionRequest(
            num_max_contacts=1, enable_contact=True
        )

    @staticmethod
    def _build_patch_objects(points, faces, patch_indices):
        """Build deterministic representative surface patches once per object."""

        faces = np.asarray(faces, dtype=np.int64)
        patches = np.asarray(patch_indices, dtype=np.int64)
        identifiers = np.unique(patches)
        result = []
        for identifier in identifiers:
            indices = np.flatnonzero(patches == identifier)
            triangles = np.asarray(points)[faces[indices]]
            area = 0.5 * np.linalg.norm(np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0]), axis=1).sum()
            result.append((int(identifier), fcl.CollisionObject(
                build_fcl_bvh_model(points, faces[indices])), indices, float(area)))
        return tuple(result)

    def _forbidden_surface_distance(self, pad, transform):
        if self._forbidden_object is None:
            return None, None
        pad.setTransform(transform)
        result = fcl.DistanceResult()
        distance = float(fcl.distance(
            self._forbidden_object, pad, self._distance_request, result))
        local_face = int(result.b1)
        face = (int(self._forbidden_object_faces[local_face])
                if 0 <= local_face < len(self._forbidden_object_faces) else None)
        return (distance if np.isfinite(distance) else None), face

    def normal(self, face_index: int) -> np.ndarray:
        if not 0 <= int(face_index) < self._face_count:
            raise ValueError("object face index is outside the registered mesh")
        return self._normals[int(face_index)]

    def query_pad(
        self, pad_name: str, transform: np.ndarray
    ) -> tuple[NearestSurface, np.ndarray, np.ndarray]:
        pad = self._pads[pad_name]
        pad.setTransform(fcl.Transform(transform[:3, :3], transform[:3, 3]))
        collision = fcl.CollisionResult()
        intersects = bool(
            fcl.collide(self._object, pad, self._collision_request, collision)
        )
        result = fcl.DistanceResult()
        distance = float(
            fcl.distance(self._object, pad, self._distance_request, result)
        )
        face_index = int(result.b1)
        surface_face_index = int(result.b2)
        if (
            not np.isfinite(distance)
            or not 0 <= face_index < self._face_count
            or not 0 <= surface_face_index < self._surface_face_count[pad_name]
            or len(result.nearest_points) != 2
        ):
            raise RuntimeError("exact contact-surface query returned an invalid witness")
        object_point = np.asarray(result.nearest_points[0], dtype=np.float64)
        pad_point = np.asarray(result.nearest_points[1], dtype=np.float64)
        surface_normal = None
        if self._task_surface_mode:
            surface_normal = (
                transform[:3, :3]
                @ self._surface_normals_local[pad_name][surface_face_index]
            )[None, :]
        nearest = NearestSurface(
            point_m=object_point[None, :],
            distance_m=np.asarray((distance,), dtype=np.float64),
            face_index=np.asarray((face_index,), dtype=np.int64),
            intersecting=intersects,
            surface_face_index=np.asarray((
                self._surface_source_faces[pad_name][surface_face_index],
            ), dtype=np.int64),
            surface_normal_m=surface_normal,
            surface_legacy_blue_pad=np.asarray((
                self._surface_legacy_blue[pad_name][surface_face_index],
            ), dtype=np.bool_),
            object_role_code=np.asarray((self._face_roles[face_index],), dtype=np.uint8),
        )
        return nearest, pad_point, self.normal(face_index)

    def query_task_surface_witnesses(
        self, pad_name: str, transform: np.ndarray, maximum_witnesses: int = 16,
    ) -> tuple[NearestSurface, np.ndarray, np.ndarray]:
        """Return finite nearest witnesses from representative registered patches."""

        if not self._task_surface_mode:
            raise RuntimeError("multi-witness queries require TASK_GRIP_SURFACE")
        limit = int(maximum_witnesses)
        if not 1 <= limit <= 16:
            raise ValueError("maximum_witnesses must lie in [1, 16]")
        full = self._pads[pad_name]
        fcl_transform = fcl.Transform(transform[:3, :3], transform[:3, 3])
        full.setTransform(fcl_transform)
        intersects = bool(fcl.collide(
            self._object, full, self._collision_request, fcl.CollisionResult()))
        forbidden_distance, forbidden_face = self._forbidden_surface_distance(
            full, fcl_transform)
        rows = []
        for patch_id, patch, surface_indices, patch_area in self._patches[pad_name]:
            patch.setTransform(fcl_transform)
            result = fcl.DistanceResult()
            distance = float(fcl.distance(
                self._allowed_object, patch, self._distance_request, result))
            object_local = int(result.b1)
            patch_local = int(result.b2)
            if (
                not np.isfinite(distance)
                or not 0 <= object_local < len(self._allowed_object_faces)
                or not 0 <= patch_local < len(surface_indices)
                or len(result.nearest_points) != 2
            ):
                continue
            surface_index = int(surface_indices[patch_local])
            rows.append((distance, patch_id, object_local, surface_index, patch_area,
                         np.asarray(result.nearest_points[0], dtype=np.float64),
                         np.asarray(result.nearest_points[1], dtype=np.float64)))
        if not rows:
            raise RuntimeError("no finite TASK_GRIP_SURFACE patch witness")
        rows.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
        finite_count = len(rows)
        rows = rows[:limit]
        object_faces = np.asarray(
            [self._allowed_object_faces[row[2]] for row in rows], dtype=np.int64)
        surface_indices = np.asarray([row[3] for row in rows], dtype=np.int64)
        nearest = NearestSurface(
            point_m=np.asarray([row[5] for row in rows]),
            distance_m=np.asarray([row[0] for row in rows]),
            face_index=object_faces,
            intersecting=intersects,
            surface_face_index=self._surface_source_faces[pad_name][surface_indices],
            surface_normal_m=(
                self._surface_normals_local[pad_name][surface_indices]
                @ transform[:3, :3].T
            ),
            surface_legacy_blue_pad=(
                self._surface_legacy_blue[pad_name][surface_indices]
            ),
            surface_patch_index=np.asarray([row[1] for row in rows], dtype=np.int64),
            surface_patch_area_m2=np.asarray([row[4] for row in rows]),
            object_role_code=self._face_roles[object_faces],
            forbidden_face_index=forbidden_face,
            forbidden_distance_m=forbidden_distance,
            registered_patch_count=len(self._patches[pad_name]),
            finite_patch_witness_count=finite_count,
        )
        return (nearest, np.asarray([row[6] for row in rows]),
                self._normals[object_faces])

    def select_task_surface_contact(
        self, pad_name: str, current: np.ndarray, moved: np.ndarray,
        object_grasp_center_m: np.ndarray, minimum_motion_m_per_phase: float,
        phase_delta: float, contact_distance_m: float,
    ) -> tuple[int, NearestSurface, np.ndarray, np.ndarray]:
        """Select a compatible allowed witness while guarding forbidden first contact."""

        from kcg_connector.grasp.carts_v2.task_grip_surface import (
            motion_compatible_with_object_witness,
        )
        full_nearest, _full_pad_point, full_normal = self.query_pad(
            pad_name, current)
        full_distance = float(np.min(full_nearest.distance_m))
        tolerance = 64.0 * np.finfo(np.float64).eps
        if (full_nearest.intersecting
                or (np.isfinite(full_distance)
                    and full_distance > contact_distance_m + tolerance)):
            return (-1, full_nearest,
                    np.asarray(full_normal, dtype=np.float64)[None, :],
                    np.asarray((-np.inf,), dtype=np.float64))
        nearest, pad_points, object_normals = self.query_task_surface_witnesses(
            pad_name, current, 16)
        pad_local = (pad_points - current[:3, 3]) @ current[:3, :3]
        moved_points = pad_local @ moved[:3, :3].T + moved[:3, 3]
        motion = (moved_points - pad_points) / float(phase_delta)
        inward = -np.einsum("ij,ij->i", motion, object_normals)
        assert nearest.surface_normal_m is not None
        compatible = motion_compatible_with_object_witness(
            pad_points, nearest.point_m, nearest.surface_normal_m,
            object_normals, motion, object_grasp_center_m,
            minimum_motion_m_per_phase)
        region = _contact_region(
            nearest, object_normals, compatible, contact_distance_m,
            self._minimum_region_area_m2)
        usable_selected = nearest_motion_compatible_index(
            nearest.distance_m, np.where(compatible, inward, -np.inf),
            minimum_motion_m_per_phase)
        selected = usable_selected if region.get("contact_region_pass") else -1
        forbidden = nearest.forbidden_distance_m
        forbidden_first = bool(
            forbidden is not None and forbidden <= contact_distance_m
            and (usable_selected < 0
                 or forbidden <= nearest.distance_m[usable_selected] + tolerance)
        )
        nearest = replace(nearest, forbidden_first_contact=forbidden_first, **region)
        return selected, nearest, object_normals, inward

    def query_points(self, points_object_m: np.ndarray) -> tuple[NearestSurface, np.ndarray]:
        if self._proximity is None:
            raise RuntimeError("dense surface fallback requires trimesh")
        points = np.asarray(points_object_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("PAD surface samples must be one finite (N,3) array")
        closest, distance, face_index = self._proximity.on_surface(points)
        face_index = np.asarray(face_index, dtype=np.int64)
        if (
            len(points) == 0
            or np.any(~np.isfinite(closest))
            or np.any(~np.isfinite(distance))
            or np.any((face_index < 0) | (face_index >= self._face_count))
        ):
            raise RuntimeError("exact object proximity returned an invalid witness")
        nearest = NearestSurface(
            point_m=np.asarray(closest, dtype=np.float64),
            distance_m=np.asarray(distance, dtype=np.float64),
            face_index=face_index,
        )
        return nearest, self._normals[face_index]


ExactPadSurfaceQuery = ExactContactSurfaceQuery


__all__ = [
    "ExactContactSurfaceQuery",
    "ExactPadSurfaceQuery",
    "LegacyMeshProximityIndex",
    "NearestSurface",
    "material_bound_object_face_normals",
    "nearest_motion_compatible_index",
]
