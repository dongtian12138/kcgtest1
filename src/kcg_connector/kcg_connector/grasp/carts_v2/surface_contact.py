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
except ImportError:
    trimesh = None

from kcg_connector.grasp.carts_v2.fast_filter import build_fcl_bvh_model


@dataclass(frozen=True)
class NearestSurface:
    point_m: np.ndarray
    distance_m: np.ndarray
    face_index: np.ndarray
    intersecting: bool = False


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


class ExactPadSurfaceQuery:
    """FCL full-mesh witness plus exact-object dense-PAD compatibility fallback."""

    def __init__(self, inputs) -> None:
        if fcl is None or trimesh is None:
            raise RuntimeError("FCL_EXACT_PAD_MESH requires python-fcl and trimesh")
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
        self._pads = {
            pad.name: fcl.CollisionObject(
                build_fcl_bvh_model(pad.points_local_m, pad.faces)
            )
            for pad in inputs.hand_contract.pads
        }
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
        if (
            not np.isfinite(distance)
            or not 0 <= face_index < self._face_count
            or len(result.nearest_points) != 2
        ):
            raise RuntimeError("exact PAD mesh query returned an invalid witness")
        object_point = np.asarray(result.nearest_points[0], dtype=np.float64)
        pad_point = np.asarray(result.nearest_points[1], dtype=np.float64)
        nearest = NearestSurface(
            point_m=object_point[None, :],
            distance_m=np.asarray((distance,), dtype=np.float64),
            face_index=np.asarray((face_index,), dtype=np.int64),
            intersecting=intersects,
        )
        return nearest, pad_point, self.normal(face_index)

    def query_points(self, points_object_m: np.ndarray) -> tuple[NearestSurface, np.ndarray]:
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


__all__ = [
    "ExactPadSurfaceQuery",
    "NearestSurface",
    "nearest_motion_compatible_index",
]
