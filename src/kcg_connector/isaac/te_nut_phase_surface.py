"""Distance to the declared static CAD surface for visual Nut phase matching.

This CPU BVH contains only source-file triangles. It has no simulator scene,
rigid-body state, or contact inputs.
"""

import numpy as np
import warp as wp


@wp.kernel
def _surface_distances(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    distances: wp.array(dtype=float),
    faces: wp.array(dtype=int),
):
    index = wp.tid()
    point = points[index]
    query = wp.mesh_query_point_no_sign(mesh_id, point, 1.0)
    if query.result:
        closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        distances[index] = wp.length(point - closest)
        faces[index] = query.face
    else:
        distances[index] = 1.0
        faces[index] = -1


class SourceTriangleSurface:
    """Closest point on continuous triangles, rather than a sampled cloud."""

    method = "SOURCE_TRIANGLE_SURFACE_DISTANCE_CPU_BVH"

    def __init__(self, vertices, faces):
        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32)
        if (vertices.ndim != 2 or vertices.shape[1] != 3
                or not np.all(np.isfinite(vertices))
                or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
                or np.any(faces < 0) or np.any(faces >= len(vertices))):
            raise ValueError("Finite source vertices and valid source triangles are required")
        self.vertices = wp.array(vertices, dtype=wp.vec3, device="cpu")
        self.indices = wp.array(faces.reshape(-1), dtype=int, device="cpu")
        self.mesh = wp.Mesh(points=self.vertices, indices=self.indices)
        self.triangle_count = len(faces)

    def query(self, points, workers=1):
        points = np.asarray(points, dtype=np.float32)
        if (points.ndim != 2 or points.shape[1] != 3
                or not np.all(np.isfinite(points))):
            raise ValueError("Visual surface queries require finite three-dimensional points")
        if not len(points):
            return np.empty(0), np.empty(0, dtype=np.int32)
        query = wp.array(points, dtype=wp.vec3, device="cpu")
        distances = wp.empty(len(points), dtype=float, device="cpu")
        faces = wp.empty(len(points), dtype=int, device="cpu")
        wp.launch(_surface_distances, dim=len(points),
                  inputs=[self.mesh.id, query, distances, faces], device="cpu")
        distances, faces = distances.numpy().astype(np.float64), faces.numpy()
        if np.any(faces < 0) or not np.all(np.isfinite(distances)):
            raise ValueError("The observed points have no finite source-surface match")
        return distances, faces
