"""Hash-bound outward normals and mature-mesh PAD/object proximity queries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import fcl
except ImportError:
    fcl = None
try:
    import trimesh
except (ImportError, TypeError):
    trimesh = None

from kcg_connector.grasp.carts_v2.fast_filter import build_fcl_bvh_model


@dataclass(frozen=True)
class NearestSurface:
    point_m: np.ndarray
    distance_m: np.ndarray
    face_index: np.ndarray
    intersecting: bool = False
    surface_face_index: np.ndarray | None = None
    surface_normal_m: np.ndarray | None = None
    surface_legacy_blue_pad: np.ndarray | None = None


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


class ExactContactSurfaceQuery:
    """FCL witness between the full object and one registered hand surface."""

    def __init__(self, inputs) -> None:
        if fcl is None:
            raise RuntimeError("FCL_EXACT_PAD_MESH requires python-fcl")
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
        self._normals = np.asarray(mesh.face_normals) * signs[:, None]
        self._face_count = len(mesh.faces)
        self._object = fcl.CollisionObject(
            build_fcl_bvh_model(mesh.vertices_m, mesh.faces)
        )
        task_surfaces = getattr(inputs, "task_grip_surfaces", None)
        geometry = (
            {
                pad.name: (
                    pad.points_local_m,
                    pad.faces,
                    np.zeros((len(pad.faces), 3), dtype=np.float64),
                    np.arange(len(pad.faces), dtype=np.int64),
                    np.ones(len(pad.faces), dtype=np.bool_),
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
                )
                for pad_name, surface in task_surfaces.items()
            }
        )
        self._task_surface_mode = task_surfaces is not None
        self._pads = {
            name: fcl.CollisionObject(build_fcl_bvh_model(points, faces))
            for name, (points, faces, _normals, _source, _legacy) in geometry.items()
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
        )
        return nearest, pad_point, self.normal(face_index)

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
    "NearestSurface",
    "nearest_motion_compatible_index",
]
