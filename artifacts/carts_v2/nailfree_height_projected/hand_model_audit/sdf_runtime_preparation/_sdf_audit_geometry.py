"""Private geometry kernels for the one-shot f1 SDF diagnostic."""

from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path
import struct

import numpy as np


def read_binary_stl(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    if len(payload) < 84:
        raise ValueError(f"STL is shorter than its binary header: {path}")
    count = struct.unpack_from("<I", payload, 80)[0]
    if len(payload) != 84 + 50 * count:
        raise ValueError(f"only provenance-bound binary STL is accepted: {path}")
    dtype = np.dtype([
        ("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")
    ])
    triangles = np.array(
        np.frombuffer(payload, dtype=dtype, count=count, offset=84)["vertices"],
        dtype=np.float64, copy=True,
    )
    if not np.all(np.isfinite(triangles)):
        raise ValueError(f"non-finite STL vertex: {path}")
    return triangles


def _canonical(triangles: np.ndarray, tolerance_m: float = 1.0e-7) -> Counter:
    quantized = np.rint(np.asarray(triangles) / tolerance_m).astype(np.int64)
    keys = []
    for triangle in quantized:
        vertices = sorted(tuple(int(value) for value in vertex) for vertex in triangle)
        keys.append(tuple(value for vertex in vertices for value in vertex))
    return Counter(keys)


def _transform_is_identity(matrix, tolerance: float = 1.0e-10) -> tuple[bool, float]:
    from pxr import Gf

    origin = np.asarray(matrix.Transform(Gf.Vec3d(0.0)), dtype=np.float64)
    columns = []
    for axis in np.eye(3):
        point = matrix.Transform(Gf.Vec3d(*(float(value) for value in axis)))
        columns.append(np.asarray(point, dtype=np.float64) - origin)
    realized = np.eye(4)
    realized[:3, :3] = np.column_stack(columns)
    realized[:3, 3] = origin
    error = float(np.max(np.abs(realized - np.eye(4))))
    return error <= tolerance, error


def find_identity_collision(stage, link: str, exact: np.ndarray, face_count: int):
    """Return the exact collider only when collider-local equals link-local."""
    from pxr import Usd, UsdGeom, UsdPhysics

    if abs(float(UsdGeom.GetStageMetersPerUnit(stage)) - 1.0) > 1.0e-12:
        raise ValueError("USD stage units are not meters")
    owners = [
        prim for prim in stage.Traverse()
        if prim.GetName() == link and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if len(owners) != 1:
        raise ValueError(f"{link}: rigid-body owner count is {len(owners)}")
    def exact_collision(prim):
        if not (prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI)):
            return False
        mesh = UsdGeom.Mesh(prim)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        if len(counts) != face_count or np.any(counts != 3) or len(indices) != 3 * face_count:
            return False
        realized = points[indices.reshape(-1, 3)]
        return _canonical(realized) == _canonical(exact)

    matches = [
        prim.GetPath() for prim in Usd.PrimRange(owners[0], Usd.TraverseInstanceProxies())
        if exact_collision(prim)
    ]
    if len(matches) != 1:
        raise ValueError(f"{link}: exact collision match count is {len(matches)}")
    collider = stage.GetPrimAtPath(matches[0])
    deinstanced = False
    if collider.IsInstanceProxy():
        instance = collider
        while not instance.IsInstance():
            instance = instance.GetParent()
        if not instance.IsValid() or not instance.IsInstanceable():
            raise ValueError(f"{link}: exact collision instance root is invalid")
        instance.SetInstanceable(False)
        collider = stage.GetPrimAtPath(matches[0])
        deinstanced = True
    if not collider.IsValid() or collider.IsInstanceProxy() or not exact_collision(collider):
        raise ValueError(f"{link}: exact collision identity changed after deinstancing")
    matrix, _ = UsdGeom.XformCache().ComputeRelativeTransform(collider, owners[0])
    identity, error = _transform_is_identity(matrix)
    if not identity:
        raise ValueError(f"collision-local differs from link-local: max error {error}")
    return owners[0], collider, {
        "stage_meters_per_unit": 1.0,
        "collision_local_from_link_identity": True,
        "collision_instance_decomposed_in_memory": deinstanced,
        "transform_max_abs_error": error,
    }


def _quaternion_matrix(quaternion) -> np.ndarray:
    w = float(quaternion.GetReal())
    x, y, z = (float(value) for value in quaternion.GetImaginary())
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("zero principal-axes quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def verify_usd_mass(link_prim, row: dict) -> dict:
    from pxr import UsdPhysics

    api = UsdPhysics.MassAPI(link_prim)
    if not api:
        raise ValueError("f1Link3 has no imported MassAPI")
    mass = float(api.GetMassAttr().Get())
    com = np.asarray(api.GetCenterOfMassAttr().Get(), dtype=np.float64)
    diagonal = np.asarray(api.GetDiagonalInertiaAttr().Get(), dtype=np.float64)
    rotation = _quaternion_matrix(api.GetPrincipalAxesAttr().Get())
    candidates = [rotation @ np.diag(diagonal) @ rotation.T,
                  rotation.T @ np.diag(diagonal) @ rotation]
    expected = np.asarray(row["mass_properties"]["new_inertia_at_com_kg_m2"])
    inertia_error = min(float(np.max(np.abs(value - expected))) for value in candidates)
    if (
        abs(mass - row["mass_properties"]["new_mass_kg"]) > 1.0e-6
        or float(np.max(np.abs(com - row["mass_properties"]["new_com_m"]))) > 1.0e-7
        or inertia_error > max(1.0e-9, 1.0e-5 * float(np.max(np.abs(expected))))
    ):
        raise ValueError("imported mass/COM/inertia differs from the bound URDF")
    return {"mass_kg": mass, "com_m": com.tolist(),
            "inertia_max_abs_error": inertia_error}


def _mesh(triangles: np.ndarray):
    import trimesh

    return trimesh.Trimesh(
        vertices=np.asarray(triangles).reshape(-1, 3),
        faces=np.arange(np.asarray(triangles).size // 3).reshape(-1, 3),
        process=True,
    )


def _closest(mesh, points: np.ndarray) -> np.ndarray:
    import trimesh

    rows = []
    for start in range(0, len(points), 131072):
        _, distance, _ = trimesh.proximity.closest_point(mesh, points[start:start + 131072])
        rows.append(distance)
    return np.concatenate(rows) if rows else np.empty(0)


def _cover(triangles: np.ndarray, radius: float) -> tuple[np.ndarray, float]:
    samples, achieved = [], 0.0
    for triangle in np.asarray(triangles):
        edges = np.linalg.norm(triangle[[1, 2, 0]] - triangle[[0, 1, 2]], axis=1)
        divisions = max(1, int(math.ceil(float(np.max(edges)) / radius)))
        achieved = max(achieved, float(np.max(edges)) / divisions)
        for first in range(divisions + 1):
            for second in range(divisions + 1 - first):
                a, b = first / divisions, second / divisions
                samples.append((1-a-b)*triangle[0] + a*triangle[1] + b*triangle[2])
    quantized = np.unique(np.rint(np.asarray(samples) / 1.0e-9).astype(np.int64), axis=0)
    return quantized.astype(np.float64) * 1.0e-9, achieved


def _component_digest(component) -> str:
    keys = sorted(_canonical(component.triangles, 1.0e-9).elements())
    digest = hashlib.sha256()
    for key in keys:
        digest.update(np.asarray(key, dtype="<i8").tobytes())
    return digest.hexdigest()


def _farthest_points(face_indices: np.ndarray, points: np.ndarray, count: int) -> np.ndarray:
    if len(points) < count:
        raise ValueError(f"only {len(points)} stable surface points remain for {count} samples")
    center = points.mean(axis=0)
    radius = np.linalg.norm(points - center, axis=1)
    selected = [int(np.lexsort((face_indices, -radius))[0])]
    distance = np.linalg.norm(points - points[selected[0]], axis=1)
    while len(selected) < count:
        distance[selected] = -np.inf
        chosen = int(np.lexsort((face_indices, -distance))[0])
        selected.append(chosen)
        distance = np.minimum(distance, np.linalg.norm(points - points[chosen], axis=1))
    return np.asarray(selected, dtype=np.int64)


def _stable_surface_samples(retained, exact, face_indices, normals, spacing, count, cohort):
    triangles = np.asarray(exact)[face_indices]
    centers = triangles.mean(axis=1)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    if np.any(twice_area <= 0.0):
        raise ValueError(f"{cohort}: degenerate source triangle")
    unit = np.asarray(normals, dtype=np.float64)
    unit /= np.linalg.norm(unit, axis=1)[:, None]
    if np.any(np.sum(unit * cross / twice_area[:, None], axis=1) < 1.0 - 1.0e-8):
        raise ValueError(f"{cohort}: bound normal differs from source triangle winding")
    edges = np.linalg.norm(triangles[:, [1, 2, 0]] - triangles[:, [0, 1, 2]], axis=2)
    edge_margin = np.min(twice_area[:, None] / (3.0 * edges), axis=1)
    plus_distance = _closest(retained, centers + spacing * unit)
    minus_distance = _closest(retained, centers - spacing * unit)
    stable = ((edge_margin >= 2.0 * spacing)
              & (np.minimum(plus_distance, minus_distance) >= 0.75 * spacing))
    eligible = np.flatnonzero(stable)
    chosen = eligible[_farthest_points(face_indices[eligible], centers[eligible], count)]
    return centers[chosen], unit[chosen], {
        "cohort": cohort, "input_face_count": int(len(face_indices)),
        "stable_candidate_count": int(len(eligible)), "selected_count": int(len(chosen)),
        "source_face_indices": face_indices[chosen].astype(int).tolist(),
        "minimum_selected_edge_margin_m": float(np.min(edge_margin[chosen])),
        "minimum_selected_other_surface_clearance_m": float(np.min(
            np.minimum(plus_distance[chosen], minus_distance[chosen]))),
    }


def _main_surface_plan(exact, task, resolution, identity):
    import trimesh

    retained = _mesh(exact)
    if len(retained.faces) != len(exact) or not np.allclose(
            retained.triangles, exact, rtol=0.0, atol=1.0e-8):
        raise ValueError("processed retained mesh no longer preserves source face indexing")
    components = trimesh.graph.connected_components(
        retained.face_adjacency, nodes=np.arange(len(retained.faces)), min_len=1)
    main_index = int(identity["component_index"])
    if main_index >= len(components):
        raise ValueError("bound main-body component index is absent")
    main_faces = np.asarray(components[main_index], dtype=np.int64)
    main = trimesh.Trimesh(vertices=retained.vertices, faces=retained.faces[main_faces], process=True)
    if (len(main_faces) != identity["face_count"]
            or _component_digest(main) != identity["triangle_digest"]):
        raise ValueError("7191-face main-body triangle identity changed")
    task_indices, task_normals = task["source_face_indices"], task["face_normals_local"]
    task_main = np.isin(task_indices, main_faces)
    if int(np.count_nonzero(task_main)) != identity["task_face_count"]:
        raise ValueError("TASK_GRIP_SURFACE membership on main body changed")
    auxiliary = []
    for key, expected in identity["auxiliary_components"].items():
        index = int(key)
        faces = np.asarray(components[index], dtype=np.int64)
        component = trimesh.Trimesh(vertices=retained.vertices, faces=retained.faces[faces], process=True)
        task_count = int(np.count_nonzero(np.isin(task_indices, faces)))
        if (len(faces) != expected["face_count"]
                or _component_digest(component) != expected["triangle_digest"] or task_count != 0):
            raise ValueError(f"auxiliary component {index} identity/semantic changed")
        auxiliary.append({"component_index": index, "face_count": int(len(faces)),
            "triangle_digest": expected["triangle_digest"], "task_grip_surface_face_count": 0,
            "role": "NON_TASK_GRIP_AUXILIARY_JOINT_HOUSING", "sign_checked": False,
            "can_rescue_main_body": False})
    spacing = float(np.max(np.ptp(np.asarray(exact).reshape(-1, 3), axis=0))) / resolution
    other_faces = np.setdiff1d(main_faces, task_indices, assume_unique=False)
    other_triangles = np.asarray(exact)[other_faces]
    other_cross = np.cross(other_triangles[:, 1]-other_triangles[:, 0],
                           other_triangles[:, 2]-other_triangles[:, 0])
    rows = [
        _stable_surface_samples(retained, exact, other_faces, other_cross, spacing,
                                identity["other_sample_count"], "MAIN_BODY_OTHER_SURFACE"),
        _stable_surface_samples(retained, exact, task_indices[task_main], task_normals[task_main],
                                spacing, identity["task_sample_count"], "MAIN_BODY_TASK_GRIP_SURFACE"),
    ]
    points, normals = np.vstack([row[0] for row in rows]), np.vstack([row[1] for row in rows])
    minimum_separation = float(np.min(np.linalg.norm(
        points[:, None, :] - points[None, :, :] + np.eye(len(points))[:, :, None] * 1.0e6,
        axis=2)))
    record = {"selection": "BOUND_MAIN_BODY_NORMAL_CROSSING_V2",
        "main_component_index": main_index, "main_component_face_count": int(len(main_faces)),
        "main_component_triangle_digest": identity["triangle_digest"],
        "main_component_watertight": bool(main.is_watertight),
        "epsilon_from_sdf_spacing_m": spacing,
        "edge_margin_requirement_m": 2.0 * spacing,
        "other_surface_clearance_requirement_m": 0.75 * spacing,
        "task_faces_on_main_component": int(np.count_nonzero(task_main)),
        "cohorts": [row[2] for row in rows], "sample_count": int(len(points)),
        "sample_bounds_local_m": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        "minimum_pairwise_sample_separation_m": minimum_separation,
        "auxiliary_components": auxiliary, "auxiliary_components_can_rescue_main_body": False}
    return points, normals, record


def _main_surface_crossing(query, exact, task, resolution, boundary, identity):
    points, normals, record = _main_surface_plan(exact, task, resolution, identity)
    epsilon = record["epsilon_from_sdf_spacing_m"]
    outside, inside = query(points + epsilon * normals), query(points - epsilon * normals)
    if not np.all(np.isfinite(outside)) or not np.all(np.isfinite(inside)):
        raise ValueError("main-body normal-crossing SDF contains non-finite values")
    outside_fail = outside <= boundary
    inside_fail = inside >= -boundary
    face_indices = sum((row["source_face_indices"] for row in record["cohorts"]), [])
    record.update({
        "outside_positive_count": int(np.count_nonzero(~outside_fail)),
        "outside_failure_count": int(np.count_nonzero(outside_fail)),
        "inside_negative_count": int(np.count_nonzero(~inside_fail)),
        "inside_failure_count": int(np.count_nonzero(inside_fail)),
        "pair_failure_count": int(np.count_nonzero(outside_fail | inside_fail)),
        "outside_sdf_minimum_m": float(np.min(outside)),
        "inside_sdf_maximum_m": float(np.max(inside)),
        "failed_source_face_indices": [int(face_indices[index]) for index in
            np.flatnonzero(outside_fail | inside_fail)],
        "pass": bool(np.all(~outside_fail) and np.all(~inside_fail)),
    })
    return record


def _removed_exclusive(source, retained, ranges, spacing, boundary):
    surface_rows, interior_rows = [], []
    for low, high in ranges:
        triangles = source[low:high + 1]
        component = _mesh(triangles)
        if not component.is_watertight or not component.is_winding_consistent:
            raise ValueError(f"removed component {low}:{high} is not closed/consistent")
        surface_rows.append(_cover(triangles, spacing)[0])
        axes = [np.arange(lo + 0.5*spacing, hi, spacing) for lo, hi in zip(*component.bounds)]
        if any(len(axis) == 0 for axis in axes):
            raise ValueError(f"removed component {low}:{high} has no interior lattice")
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        interior_rows.append(grid[component.contains(grid)])
    candidate = np.vstack(surface_rows + interior_rows)
    candidate = np.unique(np.rint(candidate / 1.0e-9).astype(np.int64), axis=0).astype(float)*1.0e-9
    exclusive = candidate[_closest(retained, candidate) > boundary]
    if not len(exclusive):
        raise ValueError("removed assembly produced no exclusive samples")
    return exclusive, {"candidate_sample_count": int(len(candidate)),
                       "exclusive_sample_count": int(len(exclusive)),
                       "lattice_spacing_m": spacing}


def _grid(query, bounds, spacing):
    low, high = bounds[0] - 2*spacing, bounds[1] + 2*spacing
    axes = [lo + np.arange(int(math.ceil((hi-lo)/spacing)) + 1)*spacing
            for lo, hi in zip(low, high)]
    count = math.prod(len(axis) for axis in axes)
    if count > 10_000_000:
        raise ValueError(f"SDF lattice exceeds bounded audit budget: {count}")
    values = np.empty(tuple(len(axis) for axis in axes), dtype=np.float32)
    yz = np.stack(np.meshgrid(axes[1], axes[2], indexing="ij"), axis=-1).reshape(-1, 2)
    for index, x_value in enumerate(axes[0]):
        points = np.column_stack((np.full(len(yz), x_value), yz))
        values[index] = query(points).reshape(len(axes[1]), len(axes[2]))
    if not np.all(np.isfinite(values)):
        raise ValueError("target SDF lattice contains non-finite values")
    return axes, values


def _zero_crossings(axes, values):
    rows = []
    for dimension in range(3):
        left_slice, right_slice = [slice(None)]*3, [slice(None)]*3
        left_slice[dimension], right_slice[dimension] = slice(None, -1), slice(1, None)
        left, right = values[tuple(left_slice)], values[tuple(right_slice)]
        mask = (left*right <= 0.0) & (np.abs(right-left) > 1.0e-12)
        indices = np.column_stack(np.nonzero(mask))
        if not len(indices):
            continue
        fraction = -left[mask] / (right[mask] - left[mask])
        points = np.column_stack([axes[axis][indices[:, axis]] for axis in range(3)])
        points[:, dimension] += fraction * (axes[dimension][1]-axes[dimension][0])
        rows.append(points)
    if not rows:
        raise ValueError("target SDF has no sampled zero crossings")
    return np.vstack(rows)


def geometry_audit(query, exact, source, ranges, resolution, cover_radius,
                   boundary, maximum_error, task_surface, main_identity) -> dict:
    retained = _mesh(exact)
    bounds = np.asarray([exact.reshape(-1, 3).min(axis=0),
                         exact.reshape(-1, 3).max(axis=0)])
    spacing = float(np.max(np.ptp(exact.reshape(-1, 3), axis=0))) / resolution
    surface, achieved = _cover(exact, cover_radius)
    surface_distance = np.abs(query(surface))
    raw_bound = float(np.max(surface_distance) + achieved)
    crossing = _main_surface_crossing(
        query, exact, task_surface, resolution, boundary, main_identity)
    removed, removed_record = _removed_exclusive(source, retained, ranges, spacing, boundary)
    removed_sdf = query(removed)
    occupied = int(np.count_nonzero(removed_sdf < -boundary))
    unresolved = int(np.count_nonzero(np.abs(removed_sdf) <= boundary))
    corners = np.asarray([[bounds[a, 0], bounds[b, 1], bounds[c, 2]]
                          for a in (0, 1) for b in (0, 1) for c in (0, 1)])
    outside = bounds.mean(axis=0) + (corners-bounds.mean(axis=0))*1.08
    outside_sdf = query(outside)
    axes, values = _grid(query, bounds, spacing)
    zeros = _zero_crossings(axes, values)
    zero_distance = _closest(retained, zeros)
    half_cell = 0.5*math.sqrt(3.0)*spacing
    cooked_bound = float(np.max(zero_distance) + half_cell)
    accepted = bool(
        raw_bound <= maximum_error and cooked_bound <= maximum_error
        and crossing["pass"] and occupied == 0 and unresolved == 0
        and np.all(outside_sdf > boundary)
    )
    return {
        "accepted": accepted, "sdf_voxel_spacing_m": spacing,
        "retained_surface": {
            "sample_count": int(len(surface)), "cover_radius_m": achieved,
            "sdf_abs_max_m": float(np.max(surface_distance)),
            "sdf_abs_p95_m": float(np.percentile(surface_distance, 95)),
            "conservative_bound_m": raw_bound,
        },
        "main_body_normal_crossing": crossing,
        "removed_exclusive": {
            **removed_record, "false_occupied_count": occupied,
            "boundary_unresolved_count": unresolved,
            "minimum_sdf_m": float(np.min(removed_sdf)),
        },
        "outside_sign": {"sample_count": len(outside),
                         "minimum_sdf_m": float(np.min(outside_sdf))},
        "cooked_surface_to_retained": {
            "zero_crossing_count": int(len(zeros)),
            "distance_max_m": float(np.max(zero_distance)),
            "distance_p95_m": float(np.percentile(zero_distance, 95)),
            "half_cell_diagonal_m": half_cell,
            "conservative_bound_m": cooked_bound, "grid_shape": list(values.shape),
        },
    }
