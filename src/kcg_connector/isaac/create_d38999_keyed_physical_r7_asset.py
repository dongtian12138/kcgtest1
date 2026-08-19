#!/usr/bin/env python3

"""Author a frozen nominal D38999 keyed physical USD asset.

The physical-model contract is the only geometry and physics-value source.
This authorer refuses to overwrite an existing asset, never computes a file
fingerprint, and does not authorize any A3 or robot-in-loop work.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kcg_connector.d38999_keyed_v2_a2_readback_result import (
    _trusted_collider_inventory,
    _trusted_family_algebra,
)
from kcg_connector.d38999_keyed_v2_physical_model_contract import (
    DEFAULT_CONTRACT_PATH,
    SUCCESSOR_ROOT_PRIM,
    WORKSPACE_ROOT,
    load_physical_model_contract,
)


QUANTUM_M = 1.0e-9
MM_TO_M = 1.0e-3
IN_TO_M = 0.0254


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a frozen nominal D38999 keyed physical asset"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONTRACT_PATH))
    parser.add_argument("--output", default=None)
    parser.add_argument("--candidate-index", type=int, default=None)
    parser.add_argument("--geometry-variant", default="nominal", choices=("nominal",))
    return parser.parse_args(argv)


def _quantized_integer(value: float) -> int:
    scaled = float(value) / QUANTUM_M
    if scaled >= 0.0:
        return int(math.floor(scaled + 0.5))
    return int(math.ceil(scaled - 0.5))


def _q(value: float) -> float:
    result = _quantized_integer(value) * QUANTUM_M
    return 0.0 if result == 0.0 else result


def _point(x: float, y: float, z: float) -> tuple[float, float, float]:
    return (_q(x), _q(y), _q(z))


def _xy(radius: float, theta_deg: float) -> tuple[float, float]:
    theta = math.radians(theta_deg)
    return radius * math.cos(theta), radius * math.sin(theta)


ANNULAR_COUNTS = [4, 4, 4, 4, 4, 4]
ANNULAR_INDICES = [
    3, 2, 1, 0,
    4, 5, 6, 7,
    0, 1, 5, 4,
    1, 2, 6, 5,
    2, 3, 7, 6,
    3, 0, 4, 7,
]
TRI_PRISM_COUNTS = [3, 3, 4, 4, 4]
TRI_PRISM_INDICES = [
    2, 1, 0,
    3, 4, 5,
    0, 1, 4, 3,
    1, 2, 5, 4,
    2, 0, 3, 5,
]
THREAD_COUNTS = [3, 3, 3, 3, 4, 4, 4, 4]
THREAD_INDICES = [
    3, 2, 0,
    2, 1, 0,
    4, 5, 7,
    5, 6, 7,
    0, 1, 5, 4,
    1, 2, 6, 5,
    2, 3, 7, 6,
    3, 0, 4, 7,
]


def _annular_wedge_points(
    *,
    center_xy: tuple[float, float] = (0.0, 0.0),
    theta0_deg: float,
    theta1_deg: float,
    z0: float,
    z1: float,
    corner_radii: Sequence[float],
) -> list[tuple[float, float, float]]:
    if len(corner_radii) != 8:
        raise ValueError("annular wedge requires eight explicit corner radii")
    cx, cy = center_xy
    thetas = (
        theta0_deg, theta0_deg, theta1_deg, theta1_deg,
        theta0_deg, theta0_deg, theta1_deg, theta1_deg,
    )
    zs = (z0, z0, z0, z0, z1, z1, z1, z1)
    output = []
    for radius, theta, z_value in zip(corner_radii, thetas, zs):
        x_value, y_value = _xy(radius, theta)
        output.append(_point(cx + x_value, cy + y_value, z_value))
    return output


def _constant_wedge_points(
    *,
    inner_radius: float,
    outer_radius: float,
    z0: float,
    z1: float,
    theta0_deg: float,
    theta1_deg: float,
    center_xy: tuple[float, float] = (0.0, 0.0),
    preserve_inner_chord_clearance: bool = False,
) -> list[tuple[float, float, float]]:
    if preserve_inner_chord_clearance:
        half_step = 0.5 * abs(theta1_deg - theta0_deg)
        authored_inner = inner_radius / math.cos(math.radians(half_step))
    else:
        authored_inner = inner_radius
    return _annular_wedge_points(
        center_xy=center_xy,
        theta0_deg=theta0_deg,
        theta1_deg=theta1_deg,
        z0=z0,
        z1=z1,
        corner_radii=(
            authored_inner, outer_radius, outer_radius, authored_inner,
            authored_inner, outer_radius, outer_radius, authored_inner,
        ),
    )


def _axial_profile_wedge_points(
    *,
    center_xy: tuple[float, float],
    theta0_deg: float,
    theta1_deg: float,
    z0: float,
    z1: float,
    inner_z0: float,
    outer_z0: float,
    inner_z1: float,
    outer_z1: float,
) -> list[tuple[float, float, float]]:
    return _annular_wedge_points(
        center_xy=center_xy,
        theta0_deg=theta0_deg,
        theta1_deg=theta1_deg,
        z0=z0,
        z1=z1,
        corner_radii=(
            inner_z0, outer_z0, outer_z0, inner_z0,
            inner_z1, outer_z1, outer_z1, inner_z1,
        ),
    )


def _polar_triangle_prism_points(
    *,
    polar_vertices: Sequence[tuple[float, float]],
    z0: float,
    z1: float,
) -> list[tuple[float, float, float]]:
    if len(polar_vertices) != 3:
        raise ValueError("polar triangle prism requires exactly three vertices")
    xy_vertices = [_xy(radius, theta_deg) for theta_deg, radius in polar_vertices]
    return [
        *[_point(x_value, y_value, z0) for x_value, y_value in xy_vertices],
        *[_point(x_value, y_value, z1) for x_value, y_value in xy_vertices],
    ]


def _radial_tangent_box_points(
    *,
    center_xy: tuple[float, float] = (0.0, 0.0),
    center_angle_deg: float,
    radial_interval: Sequence[float],
    tangential_width: float,
    axial_interval: Sequence[float],
) -> list[tuple[float, float, float]]:
    theta = math.radians(center_angle_deg)
    radial = (math.cos(theta), math.sin(theta))
    tangent = (-radial[1], radial[0])
    half_width = 0.5 * tangential_width
    radii = (
        radial_interval[0], radial_interval[1],
        radial_interval[1], radial_interval[0],
        radial_interval[0], radial_interval[1],
        radial_interval[1], radial_interval[0],
    )
    tangent_coordinates = (
        -half_width, -half_width, half_width, half_width,
        -half_width, -half_width, half_width, half_width,
    )
    zs = (axial_interval[0],) * 4 + (axial_interval[1],) * 4
    cx, cy = center_xy
    return [
        _point(
            cx + radius * radial[0] + tangent_coordinate * tangent[0],
            cy + radius * radial[1] + tangent_coordinate * tangent[1],
            z_value,
        )
        for radius, tangent_coordinate, z_value in zip(
            radii, tangent_coordinates, zs
        )
    ]


def _tangent_profile_prism(
    *,
    center_xy: tuple[float, float],
    center_angle_deg: float,
    profile_depth_radius: Sequence[Sequence[float]],
    tangential_width: float,
) -> tuple[list[tuple[float, float, float]], list[int], list[int]]:
    if len(profile_depth_radius) < 3:
        raise ValueError("tangent profile needs at least three vertices")
    theta = math.radians(center_angle_deg)
    radial = (math.cos(theta), math.sin(theta))
    tangent = (-radial[1], radial[0])
    cx, cy = center_xy
    points = []
    for tangent_coordinate in (-0.5 * tangential_width, 0.5 * tangential_width):
        for depth, radius in profile_depth_radius:
            points.append(
                _point(
                    cx + radius * radial[0] + tangent_coordinate * tangent[0],
                    cy + radius * radial[1] + tangent_coordinate * tangent[1],
                    depth,
                )
            )
    count = len(profile_depth_radius)
    counts = [count, count] + [4] * count
    indices = list(reversed(range(count))) + list(range(count, 2 * count))
    for index in range(count):
        next_index = (index + 1) % count
        indices.extend((index, next_index, next_index + count, index + count))
    return points, counts, indices


def _triangle_prism_points(
    triangle_xy: Sequence[tuple[float, float]], z0: float, z1: float
) -> list[tuple[float, float, float]]:
    if len(triangle_xy) != 3:
        raise ValueError("triangle prism requires exactly three XY vertices")
    return [
        _point(x_value, y_value, z_value)
        for z_value in (z0, z1)
        for x_value, y_value in triangle_xy
    ]


def _orient2d(
    first: tuple[int, int], second: tuple[int, int], third: tuple[int, int]
) -> int:
    return (
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )


def _incircle(
    first: tuple[int, int],
    second: tuple[int, int],
    third: tuple[int, int],
    query: tuple[int, int],
) -> int:
    ax, ay = first[0] - query[0], first[1] - query[1]
    bx, by = second[0] - query[0], second[1] - query[1]
    cx, cy = third[0] - query[0], third[1] - query[1]
    return (
        (ax * ax + ay * ay) * (bx * cy - by * cx)
        - (bx * bx + by * by) * (ax * cy - ay * cx)
        + (cx * cx + cy * cy) * (ax * by - ay * bx)
    )


def _oriented_triangle(
    triangle: Iterable[int], points_i: Sequence[tuple[int, int]]
) -> tuple[int, int, int]:
    a, b, c = tuple(triangle)
    orientation = _orient2d(points_i[a], points_i[b], points_i[c])
    if orientation == 0:
        raise ValueError("triangulation produced a zero-area triangle")
    if orientation < 0:
        b, c = c, b
    candidates = ((a, b, c), (b, c, a), (c, a, b))
    return min(candidates, key=lambda item: item[0])


def _edge(first: int, second: int) -> tuple[int, int]:
    return (first, second) if first < second else (second, first)


def _triangle_edges(triangle: Sequence[int]) -> tuple[tuple[int, int], ...]:
    return (
        _edge(triangle[0], triangle[1]),
        _edge(triangle[1], triangle[2]),
        _edge(triangle[2], triangle[0]),
    )


def _constrained_delaunay_flip(
    points_i: Sequence[tuple[int, int]],
    triangles: Iterable[Sequence[int]],
    constrained_edges: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    current = {_oriented_triangle(triangle, points_i) for triangle in triangles}
    maximum_flips = max(1000, 50 * len(current))
    flips = 0
    while True:
        edge_to_triangles: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
        for triangle in current:
            for edge in _triangle_edges(triangle):
                edge_to_triangles[edge].append(triangle)
        changed = False
        for shared_edge in sorted(edge_to_triangles):
            adjacent = edge_to_triangles[shared_edge]
            if shared_edge in constrained_edges or len(adjacent) != 2:
                continue
            a, b = shared_edge
            c = next(vertex for vertex in adjacent[0] if vertex not in shared_edge)
            d = next(vertex for vertex in adjacent[1] if vertex not in shared_edge)
            if _orient2d(points_i[a], points_i[b], points_i[c]) * _orient2d(
                points_i[a], points_i[b], points_i[d]
            ) >= 0:
                continue
            aa, bb = a, b
            if _orient2d(points_i[aa], points_i[bb], points_i[c]) < 0:
                aa, bb = bb, aa
            determinant = _incircle(
                points_i[aa], points_i[bb], points_i[c], points_i[d]
            )
            replacement_edge = _edge(c, d)
            should_flip = determinant > 0 or (
                determinant == 0 and replacement_edge < shared_edge
            )
            if not should_flip or replacement_edge in constrained_edges:
                continue
            if replacement_edge in edge_to_triangles:
                continue
            first_new = _oriented_triangle((c, d, a), points_i)
            second_new = _oriented_triangle((d, c, b), points_i)
            current.remove(adjacent[0])
            current.remove(adjacent[1])
            current.add(first_new)
            current.add(second_new)
            flips += 1
            if flips > maximum_flips:
                raise RuntimeError("constrained Delaunay edge flipping did not converge")
            changed = True
            break
        if not changed:
            break
    final_edges = {edge for triangle in current for edge in _triangle_edges(triangle)}
    missing = constrained_edges - final_edges
    if missing:
        raise ValueError(f"triangulation lost {len(missing)} constrained edges")
    return sorted(current)


def _point_inside_convex_polygon(
    point: tuple[float, float], polygon: Sequence[tuple[float, float]]
) -> bool:
    sign = 0
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        cross = (second[0] - first[0]) * (point[1] - first[1]) - (
            second[1] - first[1]
        ) * (point[0] - first[0])
        if abs(cross) <= 1.0e-20:
            continue
        current = 1 if cross > 0.0 else -1
        if sign and current != sign:
            return False
        sign = current
    return True


def _perforated_disk_triangulation(
    *,
    outer_radius: float,
    outer_segment_count: int,
    outer_phase_deg: float,
    hole_centers: Sequence[tuple[float, float]],
    hole_vertex_radius: float,
    hole_segment_count: int,
    hole_phase_deg: float,
) -> tuple[list[tuple[float, float]], list[tuple[int, int, int]]]:
    """Build the frozen no-Steiner constrained triangulation for one insert face."""

    import numpy as np
    from scipy.spatial import Delaunay

    points: list[tuple[float, float]] = []
    outer_ids = []
    for index in range(outer_segment_count):
        theta = outer_phase_deg + index * 360.0 / outer_segment_count
        x_value, y_value = _xy(outer_radius, theta)
        outer_ids.append(len(points))
        points.append((_q(x_value), _q(y_value)))
    hole_ids: list[list[int]] = []
    hole_polygons: list[list[tuple[float, float]]] = []
    for center_x, center_y in hole_centers:
        ids = []
        polygon = []
        # The contract freezes clockwise hole winding.
        for index in range(hole_segment_count):
            theta = hole_phase_deg - index * 360.0 / hole_segment_count
            dx, dy = _xy(hole_vertex_radius, theta)
            point = (_q(center_x + dx), _q(center_y + dy))
            ids.append(len(points))
            polygon.append(point)
            points.append(point)
        hole_ids.append(ids)
        hole_polygons.append(polygon)

    points_i = [(_quantized_integer(x), _quantized_integer(y)) for x, y in points]
    simplices = Delaunay(
        np.asarray(points, dtype=np.float64), qhull_options="Qbb Qc Qz Q12"
    ).simplices
    triangles = []
    for simplex in simplices:
        triangle = tuple(int(value) for value in simplex)
        centroid = (
            sum(points[index][0] for index in triangle) / 3.0,
            sum(points[index][1] for index in triangle) / 3.0,
        )
        if any(
            _point_inside_convex_polygon(centroid, polygon)
            for polygon in hole_polygons
        ):
            continue
        triangles.append(_oriented_triangle(triangle, points_i))

    constraints = {
        _edge(outer_ids[index], outer_ids[(index + 1) % len(outer_ids)])
        for index in range(len(outer_ids))
    }
    for ring in hole_ids:
        constraints.update(
            _edge(ring[index], ring[(index + 1) % len(ring)])
            for index in range(len(ring))
        )
    output = _constrained_delaunay_flip(points_i, triangles, constraints)
    expected = len(points) + 2 * len(hole_ids) - 2
    if len(output) != expected:
        raise ValueError(
            f"perforated insert triangulation produced {len(output)}, expected {expected}"
        )
    return points, output


def _line_intersection(
    line_origin: tuple[float, float],
    line_direction: tuple[float, float],
    edge_start: tuple[float, float],
    edge_end: tuple[float, float],
) -> tuple[float, float, float, float] | None:
    edge_direction = (
        edge_end[0] - edge_start[0],
        edge_end[1] - edge_start[1],
    )
    determinant = (
        line_direction[0] * (-edge_direction[1])
        - line_direction[1] * (-edge_direction[0])
    )
    if abs(determinant) <= 1.0e-18:
        return None
    rhs = (
        edge_start[0] - line_origin[0],
        edge_start[1] - line_origin[1],
    )
    line_parameter = (
        rhs[0] * (-edge_direction[1]) - rhs[1] * (-edge_direction[0])
    ) / determinant
    edge_parameter = (
        line_direction[0] * rhs[1] - line_direction[1] * rhs[0]
    ) / determinant
    return (
        line_origin[0] + line_parameter * line_direction[0],
        line_origin[1] + line_parameter * line_direction[1],
        line_parameter,
        edge_parameter,
    )


def _keyway_shell_triangulation(
    shell: Mapping[str, Any], recipe: Mapping[str, Any]
) -> tuple[list[tuple[float, float]], list[tuple[int, int, int]]]:
    outer_count = int(recipe["outer_boundary_segment_count"])
    inner_count = int(recipe["inner_boundary_segment_count"])
    if outer_count != 360 or inner_count != 360:
        raise ValueError("r7 keyway recipe requires the frozen 360/360 boundaries")
    phase = float(recipe["boundary_phase_origin_deg"])
    outer_radius = float(shell["outer_radius_m"])
    clear_radius = float(shell["clear_bore_radius_m"])
    inner_vertex_radius = clear_radius / math.cos(math.radians(0.5))
    original_outer = []
    original_inner = []
    for index in range(360):
        theta = phase - 0.5 + index
        original_outer.append(_xy(outer_radius, theta))
        original_inner.append(_xy(inner_vertex_radius, theta))

    slots = []
    remove_indices: set[int] = set()
    for key_index, (center_deg, width) in enumerate(
        zip(shell["keyway_center_angles_deg"], shell["keyway_parallel_wall_widths_m"])
    ):
        theta = math.radians(float(center_deg))
        radial = (math.cos(theta), math.sin(theta))
        tangent = (-radial[1], radial[0])
        half_width = 0.5 * float(width)
        intersections: dict[int, tuple[float, float]] = {}
        for side_sign in (-1, 1):
            origin = (side_sign * half_width * tangent[0], side_sign * half_width * tangent[1])
            candidates = []
            for edge_index, first in enumerate(original_inner):
                second = original_inner[(edge_index + 1) % 360]
                intersection = _line_intersection(origin, radial, first, second)
                if intersection is None:
                    continue
                x_value, y_value, line_parameter, edge_parameter = intersection
                if line_parameter > 0.0 and -1.0e-12 <= edge_parameter <= 1.0 + 1.0e-12:
                    candidates.append((line_parameter, edge_index, x_value, y_value))
            if len(candidates) != 1:
                raise ValueError(
                    f"keyway {key_index} side {side_sign} has {len(candidates)} inner intersections"
                )
            _, edge_index, x_value, y_value = candidates[0]
            intersections[side_sign] = (x_value, y_value)
            # The selected edge is split; removal is decided from the full opening below.
            if edge_index < 0:
                raise AssertionError("unreachable keyway edge index")

        outer_line_parameter = math.sqrt(
            float(shell["keyway_slot_end_radius_m"]) ** 2 - half_width**2
        )
        outer_negative = (
            outer_line_parameter * radial[0] - half_width * tangent[0],
            outer_line_parameter * radial[1] - half_width * tangent[1],
        )
        outer_positive = (
            outer_line_parameter * radial[0] + half_width * tangent[0],
            outer_line_parameter * radial[1] + half_width * tangent[1],
        )
        slots.append(
            [
                intersections[-1],
                outer_negative,
                outer_positive,
                intersections[1],
            ]
        )
        for vertex_index, point in enumerate(original_inner):
            radial_coordinate = point[0] * radial[0] + point[1] * radial[1]
            tangent_coordinate = point[0] * tangent[0] + point[1] * tangent[1]
            if radial_coordinate > 0.0 and abs(tangent_coordinate) <= half_width:
                remove_indices.add(vertex_index)

    declared_removed = sum(shell["inner_boundary_vertices_removed_by_keyway_in_declared_key_order"])
    if len(remove_indices) != declared_removed:
        raise ValueError(
            f"keyway recipe removed {len(remove_indices)} inner vertices, expected {declared_removed}"
        )

    points: list[tuple[float, float]] = [(_q(x), _q(y)) for x, y in original_outer]
    retained_ids: dict[int, int] = {}
    for index, (x_value, y_value) in enumerate(original_inner):
        if index not in remove_indices:
            retained_ids[index] = len(points)
            points.append((_q(x_value), _q(y_value)))
    inserted_ids: list[list[int]] = []
    for slot in slots:
        ids = []
        for x_value, y_value in slot:
            ids.append(len(points))
            points.append((_q(x_value), _q(y_value)))
        inserted_ids.append(ids)
    if len(points) != 360 + int(shell["modified_inner_boundary_vertex_count"]):
        raise ValueError("modified keyway boundary vertex count does not close")

    boundary_events: list[tuple[float, int]] = []
    for original_index, vertex_id in retained_ids.items():
        angle = (phase - 0.5 + original_index) % 360.0
        boundary_events.append((angle, vertex_id))
    for ids in inserted_ids:
        for vertex_id in ids:
            x_value, y_value = points[vertex_id]
            angle = math.degrees(math.atan2(y_value, x_value)) % 360.0
            boundary_events.append((angle, vertex_id))
    boundary_events.sort(key=lambda item: (item[0], item[1]))
    inner_loop = [vertex_id for _, vertex_id in boundary_events]
    inner_angles = [angle for angle, _ in boundary_events]
    if len(inner_loop) != int(shell["modified_inner_boundary_vertex_count"]):
        raise ValueError("modified keyway loop count differs")

    outer_loop = list(range(360))
    outer_angles = [float(index) for index in range(360)]
    # Start the inner walk at the final point before angle zero, unwrapped negative.
    start_inner = max(
        range(len(inner_loop)),
        key=lambda index: inner_angles[index] if inner_angles[index] <= 360.0 else -math.inf,
    )
    rotated_inner = inner_loop[start_inner:] + inner_loop[:start_inner]
    rotated_angles = inner_angles[start_inner:] + inner_angles[:start_inner]
    if rotated_angles[0] > 0.0:
        rotated_angles[0] -= 360.0
    # If only the first point was moved negative, keep all following angles ahead of it.
    for index in range(1, len(rotated_angles)):
        while rotated_angles[index] <= rotated_angles[index - 1]:
            rotated_angles[index] += 360.0

    outer_extended = outer_loop + [outer_loop[0]]
    outer_angles_extended = outer_angles + [360.0]
    inner_extended = rotated_inner + [rotated_inner[0]]
    inner_angles_extended = rotated_angles + [rotated_angles[0] + 360.0]
    outer_index = 0
    inner_index = 0
    triangles = []
    while outer_index < len(outer_loop) or inner_index < len(rotated_inner):
        next_outer = (
            outer_angles_extended[outer_index + 1]
            if outer_index < len(outer_loop)
            else math.inf
        )
        next_inner = (
            inner_angles_extended[inner_index + 1]
            if inner_index < len(rotated_inner)
            else math.inf
        )
        if next_outer <= next_inner:
            triangles.append(
                (outer_extended[outer_index], outer_extended[outer_index + 1], inner_extended[inner_index])
            )
            outer_index += 1
        else:
            triangles.append(
                (outer_extended[outer_index], inner_extended[inner_index + 1], inner_extended[inner_index])
            )
            inner_index += 1
    points_i = [(_quantized_integer(x), _quantized_integer(y)) for x, y in points]
    constraints = {
        _edge(outer_loop[index], outer_loop[(index + 1) % len(outer_loop)])
        for index in range(len(outer_loop))
    }
    constraints.update(
        _edge(inner_loop[index], inner_loop[(index + 1) % len(inner_loop)])
        for index in range(len(inner_loop))
    )
    output = _constrained_delaunay_flip(points_i, triangles, constraints)
    expected = int(shell["expected_CDT_triangle_and_convex_prism_count"])
    if len(output) != expected:
        raise ValueError(
            f"keyway triangulation produced {len(output)}, expected {expected}"
        )
    return points, output


def _apply_unknown_schema(prim: Any, schema_name: str, Sdf: Any) -> None:
    schemas = list(prim.GetAppliedSchemas())
    metadata = prim.GetMetadata("apiSchemas")
    if metadata is not None:
        schemas.extend(str(item) for item in metadata.GetAddedOrExplicitItems())
    if schema_name not in schemas:
        schemas.append(schema_name)
    unique = list(dict.fromkeys(schemas))
    prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(unique))


def _custom_string(prim: Any, name: str, value: str, Sdf: Any) -> None:
    prim.CreateAttribute(name, Sdf.ValueTypeNames.String, custom=True).Set(str(value))


def _custom_bool(prim: Any, name: str, value: bool, Sdf: Any) -> None:
    prim.CreateAttribute(name, Sdf.ValueTypeNames.Bool, custom=True).Set(bool(value))


def _ensure_xform(stage: Any, path: str, UsdGeom: Any) -> Any:
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        return prim
    parent = str(Path(path).parent)
    if parent not in (".", "/"):
        _ensure_xform(stage, parent, UsdGeom)
    return UsdGeom.Xform.Define(stage, path).GetPrim()


def _extent(points: Sequence[Sequence[float]], Gf: Any) -> list[Any]:
    lows = [min(point[axis] for point in points) for axis in range(3)]
    highs = [max(point[axis] for point in points) for axis in range(3)]
    return [Gf.Vec3f(*lows), Gf.Vec3f(*highs)]


class _AssetAuthorer:
    def __init__(
        self,
        *,
        stage: Any,
        model: Any,
        Gf: Any,
        Sdf: Any,
        UsdGeom: Any,
        UsdPhysics: Any,
        UsdShade: Any,
    ) -> None:
        self.stage = stage
        self.model = model
        self.Gf = Gf
        self.Sdf = Sdf
        self.UsdGeom = UsdGeom
        self.UsdPhysics = UsdPhysics
        self.UsdShade = UsdShade
        self.document = model.document
        self.blueprint = self.document["a2_collision_authoring_blueprint"]
        self.families = self.blueprint["filtering"]["primitive_family_definitions"]
        self.materials: dict[tuple[str, str], Any] = {}
        self.family_paths: dict[str, list[str]] = defaultdict(list)
        self.authored_colliders: set[str] = set()
        self.colors = {
            "anti_decoupling_detent": (0.72, 0.34, 0.10),
            "coupling_bearing_and_shoulder": (0.42, 0.46, 0.52),
            "coupling_nut_outer_grip": (0.28, 0.31, 0.36),
            "coupling_thread": (0.54, 0.57, 0.62),
            "fixture_and_receptacle": (0.26, 0.43, 0.64),
            "interfacial_pin_barrier": (0.83, 0.24, 0.16),
            "peripheral_seal": (0.48, 0.05, 0.07),
            "pin_and_socket": (0.78, 0.57, 0.14),
            "plug_shell_and_keys": (0.54, 0.56, 0.60),
            "spring_finger": (0.70, 0.42, 0.16),
        }

    def _material(self, material_role: str, response_role: str) -> Any:
        identity = (material_role, response_role)
        if identity in self.materials:
            return self.materials[identity]
        path = f"{SUCCESSOR_ROOT_PRIM}/Materials/{material_role}__{response_role}"
        _ensure_xform(self.stage, str(Path(path).parent), self.UsdGeom)
        material = self.UsdShade.Material.Define(self.stage, path)
        role = self.document["material_roles"]["roles"][material_role]
        response = self.document["material_roles"]["response_roles"][response_role]
        standard = self.UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        standard.CreateStaticFrictionAttr(float(role["static_friction"]))
        standard.CreateDynamicFrictionAttr(float(role["dynamic_friction"]))
        standard.CreateRestitutionAttr(float(role["restitution"]))
        prim = material.GetPrim()
        _apply_unknown_schema(prim, "PhysxMaterialAPI", self.Sdf)
        combine = self.document["solver_profile"]["authored_attribute_contract"]["materials"]
        for name, value in combine.items():
            prim.CreateAttribute(name, self.Sdf.ValueTypeNames.Token, custom=False).Set(value)
        if response["class"] == "compliant":
            stiffness = float(response["nominal_stiffness_n_m"])
            damping = float(response["nominal_damping_n_s_m"])
            acceleration = bool(response["accelerationSpring"])
        else:
            stiffness = float(response["compliant_stiffness_n_m"])
            damping = float(response["compliant_damping_n_s_m"])
            acceleration = False
        prim.CreateAttribute(
            "physxMaterial:compliantContactStiffness",
            self.Sdf.ValueTypeNames.Float,
            custom=False,
        ).Set(stiffness)
        prim.CreateAttribute(
            "physxMaterial:compliantContactDamping",
            self.Sdf.ValueTypeNames.Float,
            custom=False,
        ).Set(damping)
        prim.CreateAttribute(
            "physxMaterial:compliantContactAccelerationSpring",
            self.Sdf.ValueTypeNames.Bool,
            custom=False,
        ).Set(acceleration)
        _custom_string(prim, "kcg:materialRole", material_role, self.Sdf)
        _custom_string(prim, "kcg:responseRole", response_role, self.Sdf)
        self.materials[identity] = material
        return material

    def _mark_collider(
        self,
        prim: Any,
        *,
        family: str,
        recipe_id: str,
        topology_signature: str,
        analytic: bool = False,
    ) -> None:
        if family not in self.families:
            raise ValueError(f"unknown primitive family {family}")
        path = str(prim.GetPath())
        if path in self.authored_colliders:
            raise ValueError(f"duplicate collider prim {path}")
        family_contract = self.families[family]
        collision = self.UsdPhysics.CollisionAPI.Apply(prim)
        collision.CreateCollisionEnabledAttr(True)
        if not analytic:
            mesh_collision = self.UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_collision.CreateApproximationAttr("convexHull")
            cooking = self.document["convex_cooking_representation"]
            _apply_unknown_schema(prim, "PhysxConvexHullCollisionAPI", self.Sdf)
            prim.CreateAttribute(
                "physxConvexHullCollision:minThickness",
                self.Sdf.ValueTypeNames.Float,
                custom=False,
            ).Set(
                float(
                    cooking[
                        "physxConvexHullCollision:minThickness_local_units"
                    ]
                )
            )
        _apply_unknown_schema(prim, "PhysxCollisionAPI", self.Sdf)
        offsets = self.document["solver_profile"]["authored_attribute_contract"][
            "fine_connector_colliders"
        ]
        prim.CreateAttribute(
            "physxCollision:contactOffset", self.Sdf.ValueTypeNames.Float, custom=False
        ).Set(float(offsets["physxCollision:contactOffset"]))
        prim.CreateAttribute(
            "physxCollision:restOffset", self.Sdf.ValueTypeNames.Float, custom=False
        ).Set(float(offsets["physxCollision:restOffset"]))
        material = self._material(
            family_contract["material_role"], family_contract["response_role"]
        )
        self.UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, materialPurpose="physics"
        )
        _custom_string(prim, "kcg:primitiveFamily", family, self.Sdf)
        _custom_string(
            prim, "kcg:materialRole", family_contract["material_role"], self.Sdf
        )
        _custom_string(
            prim, "kcg:responseRole", family_contract["response_role"], self.Sdf
        )
        _custom_string(prim, "kcg:offsetClass", "fine_connector", self.Sdf)
        _custom_string(prim, "kcg:primitiveRecipeId", recipe_id, self.Sdf)
        _custom_string(
            prim,
            "kcg:cookingRepresentation",
            (
                "analytic_not_applicable"
                if analytic
                else self.document["convex_cooking_representation"][
                    "representation_id"
                ]
            ),
            self.Sdf,
        )
        _custom_string(prim, "kcg:topologySignature", topology_signature, self.Sdf)
        _custom_bool(prim, "kcg:closedManifold", True, self.Sdf)
        _custom_bool(prim, "kcg:positiveVolume", True, self.Sdf)
        _custom_bool(prim, "kcg:convex", True, self.Sdf)
        self.family_paths[family].append(path)
        self.authored_colliders.add(path)

    def mesh(
        self,
        *,
        family: str,
        path: str,
        points: Sequence[Sequence[float]],
        counts: Sequence[int],
        indices: Sequence[int],
        recipe_id: str,
    ) -> None:
        _ensure_xform(self.stage, str(Path(path).parent), self.UsdGeom)
        mesh = self.UsdGeom.Mesh.Define(self.stage, path)
        cooking = self.document["convex_cooking_representation"]
        point_multiplier = float(
            cooking["authored_mesh_point_multiplier_from_blueprint"]
        )
        raw_points = [
            tuple(float(value) * point_multiplier for value in point)
            for point in points
        ]
        mesh.CreatePointsAttr([self.Gf.Vec3f(*point) for point in raw_points])
        mesh.CreateFaceVertexCountsAttr(list(counts))
        mesh.CreateFaceVertexIndicesAttr(list(indices))
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateOrientationAttr("rightHanded")
        mesh.CreateExtentAttr(_extent(raw_points, self.Gf))
        scale = cooking["mesh_uniform_scale_xyz"]
        self.UsdGeom.Xformable(mesh).AddScaleOp().Set(self.Gf.Vec3f(*scale))
        material_role = self.families[family]["material_role"]
        mesh.CreateDisplayColorAttr([self.Gf.Vec3f(*self.colors[material_role])])
        signature = (
            f"recipe={recipe_id};points={len(points)};faces={len(counts)};"
            "closed=true;convex=true"
        )
        self._mark_collider(
            mesh.GetPrim(),
            family=family,
            recipe_id=recipe_id,
            topology_signature=signature,
        )

    def cylinder(
        self,
        *,
        family: str,
        path: str,
        center: tuple[float, float, float],
        radius: float,
        height: float,
    ) -> None:
        _ensure_xform(self.stage, str(Path(path).parent), self.UsdGeom)
        cylinder = self.UsdGeom.Cylinder.Define(self.stage, path)
        cylinder.CreateAxisAttr("Z")
        cylinder.CreateRadiusAttr(float(radius))
        cylinder.CreateHeightAttr(float(height))
        cylinder.CreateExtentAttr(
            [
                self.Gf.Vec3f(-radius, -radius, -0.5 * height),
                self.Gf.Vec3f(radius, radius, 0.5 * height),
            ]
        )
        self.UsdGeom.Xformable(cylinder).AddTranslateOp().Set(
            self.Gf.Vec3d(*center)
        )
        material_role = self.families[family]["material_role"]
        cylinder.CreateDisplayColorAttr([self.Gf.Vec3f(*self.colors[material_role])])
        self._mark_collider(
            cylinder.GetPrim(),
            family=family,
            recipe_id="analytic_cylinder_v1",
            topology_signature="analytic=cylinder;axis=Z;closed=true;convex=true",
            analytic=True,
        )

    def sphere(
        self,
        *,
        family: str,
        path: str,
        center: tuple[float, float, float],
        radius: float,
    ) -> None:
        _ensure_xform(self.stage, str(Path(path).parent), self.UsdGeom)
        sphere = self.UsdGeom.Sphere.Define(self.stage, path)
        sphere.CreateRadiusAttr(float(radius))
        sphere.CreateExtentAttr(
            [
                self.Gf.Vec3f(-radius, -radius, -radius),
                self.Gf.Vec3f(radius, radius, radius),
            ]
        )
        self.UsdGeom.Xformable(sphere).AddTranslateOp().Set(
            self.Gf.Vec3d(*center)
        )
        material_role = self.families[family]["material_role"]
        sphere.CreateDisplayColorAttr([self.Gf.Vec3f(*self.colors[material_role])])
        self._mark_collider(
            sphere.GetPrim(),
            family=family,
            recipe_id="analytic_sphere_v1",
            topology_signature="analytic=sphere;closed=true;convex=true",
            analytic=True,
        )

    def capsule(
        self,
        *,
        family: str,
        path: str,
        center: tuple[float, float, float],
        radius: float,
        height: float,
        direction: tuple[float, float, float],
        recipe_id: str = "analytic_capsule_helix_chord_v1",
    ) -> None:
        _ensure_xform(self.stage, str(Path(path).parent), self.UsdGeom)
        capsule = self.UsdGeom.Capsule.Define(self.stage, path)
        capsule.CreateAxisAttr("X")
        capsule.CreateRadiusAttr(float(radius))
        capsule.CreateHeightAttr(float(height))
        half_extent = 0.5 * height + radius
        capsule.CreateExtentAttr(
            [
                self.Gf.Vec3f(-half_extent, -radius, -radius),
                self.Gf.Vec3f(half_extent, radius, radius),
            ]
        )
        xformable = self.UsdGeom.Xformable(capsule)
        xformable.AddTranslateOp().Set(self.Gf.Vec3d(*center))
        norm = math.sqrt(sum(float(value) ** 2 for value in direction))
        if norm <= 0.0:
            raise ValueError("capsule direction must be nonzero")
        vx, vy, vz = (float(value) / norm for value in direction)
        w = math.sqrt(max(0.0, 0.5 * (1.0 + vx)))
        if w <= 1.0e-12:
            quaternion = self.Gf.Quatf(0.0, self.Gf.Vec3f(0.0, 0.0, 1.0))
        else:
            quaternion = self.Gf.Quatf(
                w,
                self.Gf.Vec3f(0.0, -vz / (2.0 * w), vy / (2.0 * w)),
            )
        xformable.AddOrientOp().Set(quaternion)
        material_role = self.families[family]["material_role"]
        capsule.CreateDisplayColorAttr([self.Gf.Vec3f(*self.colors[material_role])])
        self._mark_collider(
            capsule.GetPrim(),
            family=family,
            recipe_id=recipe_id,
            topology_signature=(
                f"analytic=capsule;axis_direction={direction};closed=true;convex=true"
            ),
            analytic=True,
        )

    def author_identity_and_rigid_bodies(self) -> None:
        world = self.UsdGeom.Xform.Define(self.stage, "/World")
        self.stage.SetDefaultPrim(world.GetPrim())
        root = self.UsdGeom.Xform.Define(self.stage, SUCCESSOR_ROOT_PRIM)
        identity = self.document["identity"]
        for name, value in {
            "schemaVersion": identity["successor_schema"],
            "assetRevision": identity["successor_revision"],
            "pairModelId": identity["pair_model_id"],
            "loosePlugModelId": identity["loose_plug_model_id"],
            "fixedReceptacleModelId": identity["fixed_receptacle_model_id"],
            "geometryVariant": "nominal",
            "fidelity": "public_spec_geometry_with_frozen_force_transmitting_simulation_proxies",
        }.items():
            _custom_string(root.GetPrim(), f"kcg:{name}", value, self.Sdf)
        _custom_bool(root.GetPrim(), "kcg:hardwareAuthoritative", False, self.Sdf)

        self.UsdGeom.Xform.Define(self.stage, f"{SUCCESSOR_ROOT_PRIM}/FixedReceptacle")
        self.UsdGeom.Xform.Define(self.stage, f"{SUCCESSOR_ROOT_PRIM}/LoosePlug")
        self.UsdGeom.Xform.Define(
            self.stage, f"{SUCCESSOR_ROOT_PRIM}/LoosePlug/BodyAssembly"
        )
        self.UsdGeom.Xform.Define(
            self.stage, f"{SUCCESSOR_ROOT_PRIM}/LoosePlug/CouplingNut"
        )
        mass_contract = self.document["realized_robot_hand_fixture_blueprint"][
            "fixture_load_path"
        ]["connector_body_mass_derivation"]
        owner_paths = {
            "FixedReceptacle": f"{SUCCESSOR_ROOT_PRIM}/FixedReceptacle",
            "BodyAssembly": f"{SUCCESSOR_ROOT_PRIM}/LoosePlug/BodyAssembly",
            "CouplingNut": f"{SUCCESSOR_ROOT_PRIM}/LoosePlug/CouplingNut",
        }
        rigid_contract = self.document["solver_profile"]["authored_attribute_contract"][
            "dynamic_rigid_bodies"
        ]
        principal = mass_contract["principal_axes_quaternion_wxyz"]
        for owner, path in owner_paths.items():
            prim = self.stage.GetPrimAtPath(path)
            rigid = self.UsdPhysics.RigidBodyAPI.Apply(prim)
            rigid.CreateRigidBodyEnabledAttr(True)
            rigid.CreateKinematicEnabledAttr(False)
            mass_values = mass_contract["bodies"][owner]
            mass_api = self.UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr(float(mass_values["mass_kg"]))
            mass_api.CreateCenterOfMassAttr(
                self.Gf.Vec3f(*mass_values["local_com_m"])
            )
            mass_api.CreateDiagonalInertiaAttr(
                self.Gf.Vec3f(*mass_values["diagonal_inertia_kg_m2"])
            )
            mass_api.CreatePrincipalAxesAttr(
                self.Gf.Quatf(
                    float(principal[0]),
                    self.Gf.Vec3f(*[float(value) for value in principal[1:]]),
                )
            )
            _apply_unknown_schema(prim, "PhysxRigidBodyAPI", self.Sdf)
            prim.CreateAttribute(
                "physxRigidBody:enableCCD", self.Sdf.ValueTypeNames.Bool, custom=False
            ).Set(bool(rigid_contract["physxRigidBody:enableCCD"]))
            prim.CreateAttribute(
                "physxRigidBody:maxDepenetrationVelocity",
                self.Sdf.ValueTypeNames.Float,
                custom=False,
            ).Set(float(rigid_contract["physxRigidBody:maxDepenetrationVelocity"]))
            for field in (
                "physxRigidBody:solverPositionIterationCount",
                "physxRigidBody:solverVelocityIterationCount",
            ):
                prim.CreateAttribute(field, self.Sdf.ValueTypeNames.Int, custom=False).Set(
                    int(rigid_contract[field])
                )
            _custom_string(prim, "kcg:rigidOwner", owner, self.Sdf)

    def author_joint(self) -> None:
        contract = self.document["solver_profile"]["authored_attribute_contract"][
            "nut_body_D6_joint"
        ]
        path = f"{SUCCESSOR_ROOT_PRIM}/LoosePlug/CouplingNutJoint"
        joint = self.UsdPhysics.Joint.Define(self.stage, path)
        joint.CreateJointEnabledAttr(bool(contract["physics:jointEnabled"]))
        joint.CreateCollisionEnabledAttr(bool(contract["physics:collisionEnabled"]))
        joint.CreateBody0Rel().SetTargets([self.Sdf.Path(contract["physics:body0"])])
        joint.CreateBody1Rel().SetTargets([self.Sdf.Path(contract["physics:body1"])])
        joint.CreateLocalPos0Attr(self.Gf.Vec3f(*contract["physics:localPos0"]))
        joint.CreateLocalPos1Attr(self.Gf.Vec3f(*contract["physics:localPos1"]))
        q0 = contract["physics:localRot0_wxyz"]
        q1 = contract["physics:localRot1_wxyz"]
        joint.CreateLocalRot0Attr(self.Gf.Quatf(q0[0], self.Gf.Vec3f(*q0[1:])))
        joint.CreateLocalRot1Attr(self.Gf.Quatf(q1[0], self.Gf.Vec3f(*q1[1:])))
        for axis in ("transX", "transY", "transZ", "rotX", "rotY"):
            limit = self.UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
            limit.CreateLowAttr(float(contract[f"limit:{axis}:physics:low"]))
            limit.CreateHighAttr(float(contract[f"limit:{axis}:physics:high"]))
        _custom_string(
            joint.GetPrim(), "kcg:jointRole", "passive_coupling_nut_D6", self.Sdf
        )

    def _contacts(self, *, plug_local: bool) -> list[tuple[str, float, float]]:
        output = []
        for label, source_x, source_y in self.blueprint["contact_layout"][
            "positions_in_exactly"
        ]:
            x_value = _q(float(source_x) * IN_TO_M)
            y_value = _q(float(source_y) * IN_TO_M)
            output.append((str(label), x_value, -y_value if plug_local else y_value))
        if len(output) != 61:
            raise ValueError("contact layout no longer contains 61 positions")
        return output

    def author_shells_and_keying(self) -> None:
        section = self.blueprint["connector_shells_and_keying"]
        annular = section["annular_wedge_recipe"]
        step = float(annular["angular_step_deg"])
        phase = float(annular["phase_origin_deg"])

        plug = section["plug_mating_shell"]
        for index in range(int(annular["angular_segment_count"])):
            center = phase + index * step
            points = _constant_wedge_points(
                inner_radius=float(plug["radial_interval_m"][0]),
                outer_radius=float(plug["radial_interval_m"][1]),
                z0=float(plug["local_depth_interval_m"][0]),
                z1=float(plug["local_depth_interval_m"][1]),
                theta0_deg=center - 0.5 * step,
                theta1_deg=center + 0.5 * step,
                preserve_inner_chord_clearance=True,
            )
            self.mesh(
                family="plug_mating_shell_360",
                path=plug["piece_path_template"].format(segment_index=index),
                points=points,
                counts=ANNULAR_COUNTS,
                indices=ANNULAR_INDICES,
                recipe_id="annular_wedge_8v_v1",
            )

        carrier = section["receptacle_thread_carrier"]
        for index in range(int(annular["angular_segment_count"])):
            center = phase + index * step
            points = _constant_wedge_points(
                inner_radius=float(carrier["radial_interval_m"][0]),
                outer_radius=float(carrier["radial_interval_m"][1]),
                z0=float(carrier["local_z_interval_m"][0]),
                z1=float(carrier["local_z_interval_m"][1]),
                theta0_deg=center - 0.5 * step,
                theta1_deg=center + 0.5 * step,
                preserve_inner_chord_clearance=True,
            )
            self.mesh(
                family="receptacle_thread_carrier_360",
                path=carrier["piece_path_template"].format(segment_index=index),
                points=points,
                counts=ANNULAR_COUNTS,
                indices=ANNULAR_INDICES,
                recipe_id="annular_wedge_8v_v1",
            )

        keyway = section["receptacle_keyway_shell"]
        keyway_recipe = self.blueprint["canonical_primitive_recipes"]["keyway_pslg_v1"]
        keyway_points, keyway_triangles = _keyway_shell_triangulation(
            keyway, keyway_recipe
        )
        for piece_index, triangle in enumerate(keyway_triangles):
            points = _triangle_prism_points(
                [keyway_points[index] for index in triangle],
                float(keyway["local_z_interval_m"][0]),
                float(keyway["local_z_interval_m"][1]),
            )
            self.mesh(
                family="receptacle_keyway_shell_prisms",
                path=keyway["prism_path_template"].format(piece_index=piece_index),
                points=points,
                counts=TRI_PRISM_COUNTS,
                indices=TRI_PRISM_INDICES,
                recipe_id="planar_triangle_z_prism_6v_v1",
            )

        keys = section["plug_keys"]
        for key_index, (angle, width) in enumerate(
            zip(keys["center_angles_deg"], keys["tangent_frame_widths_m"])
        ):
            nose_points, nose_counts, nose_indices = _tangent_profile_prism(
                center_xy=(0.0, 0.0),
                center_angle_deg=float(angle),
                profile_depth_radius=keys["nose_depth_radial_profile_points_m"],
                tangential_width=float(width),
            )
            self.mesh(
                family="plug_keys_10",
                path=keys["nose_path_template"].format(key_index=key_index),
                points=nose_points,
                counts=nose_counts,
                indices=nose_indices,
                recipe_id="tangent_profile_prism_v1",
            )
            full_points = _radial_tangent_box_points(
                center_angle_deg=float(angle),
                radial_interval=keys["radial_interval_m"],
                tangential_width=float(width),
                axial_interval=(
                    float(keys["full_radial_section_starts_local_z_m"]),
                    float(keys["local_depth_interval_m"][1]),
                ),
            )
            self.mesh(
                family="plug_keys_10",
                path=keys["full_section_path_template"].format(key_index=key_index),
                points=full_points,
                counts=ANNULAR_COUNTS,
                indices=ANNULAR_INDICES,
                recipe_id="radial_tangent_box_8v_v1",
            )

        nut = section["coupling_nut"]
        nut_step = float(nut["angular_step_deg"])
        for family, template, axial, radial in (
            (
                "coupling_nut_front_envelope_96",
                nut["front_thread_carrier_piece_path_template"],
                nut["front_thread_carrier_local_z_interval_m"],
                nut["front_thread_carrier_radial_interval_m"],
            ),
            (
                "coupling_nut_rear_envelope_96",
                nut["rear_grip_piece_path_template"],
                nut["rear_grip_local_z_interval_m"],
                nut["rear_grip_radial_interval_m"],
            ),
        ):
            for index in range(int(nut["wedge_segment_count"])):
                center = float(nut["phase_origin_deg"]) + index * nut_step
                self.mesh(
                    family=family,
                    path=template.format(segment_index=index),
                    points=_constant_wedge_points(
                        inner_radius=float(radial[0]),
                        outer_radius=float(radial[1]),
                        z0=float(axial[0]),
                        z1=float(axial[1]),
                        theta0_deg=center - 0.5 * nut_step,
                        theta1_deg=center + 0.5 * nut_step,
                    ),
                    counts=ANNULAR_COUNTS,
                    indices=ANNULAR_INDICES,
                    recipe_id="annular_wedge_8v_v1",
                )

        rear = section["body_assembly_rear_body"]
        rear_step = float(rear["angular_step_deg"])
        for band in rear["local_z_profile_bands"]:
            for index in range(int(rear["wedge_segment_count"])):
                center = float(rear["phase_origin_deg"]) + index * rear_step
                self.mesh(
                    family="body_assembly_rear_body_288",
                    path=rear["piece_path_template"].format(
                        profile_band_index=band["profile_band_index"],
                        segment_index=index,
                    ),
                    points=_constant_wedge_points(
                        inner_radius=float(band["inner_radius_m"]),
                        outer_radius=float(band["outer_radius_m"]),
                        z0=float(band["z_start_m"]),
                        z1=float(band["z_end_m"]),
                        theta0_deg=center - 0.5 * rear_step,
                        theta1_deg=center + 0.5 * rear_step,
                    ),
                    counts=ANNULAR_COUNTS,
                    indices=ANNULAR_INDICES,
                    recipe_id="annular_wedge_8v_v1",
                )

    def author_thread(self) -> None:
        section = self.blueprint["thread"]
        inner_radius, outer_radius = [float(value) for value in section["rail_radial_interval_m"]]
        thickness = float(section["rail_axial_thickness_m"])
        for start_index, phase in enumerate(section["start_phases_deg"]):
            for segment_index in range(int(section["segments_per_start"])):
                theta0 = float(phase) + segment_index * float(section["segment_angle_deg"])
                theta1 = float(phase) + (segment_index + 1) * float(section["segment_angle_deg"])
                contact_z0 = 0.00912 + 0.00762 * segment_index / 360.0
                contact_z1 = 0.00912 + 0.00762 * (segment_index + 1) / 360.0
                if section["rail_piece_shape"] == (
                    "analytic_capsule_chain_along_original_helix_chords"
                ):
                    radius = float(section["rail_capsule_radius_m"])
                    centerline_radius = float(
                        section["rail_capsule_centerline_radius_m"]
                    )
                    axial_offset = float(
                        section["rail_capsule_centerline_axial_offset_m"]
                    )
                    x0, y0 = _xy(centerline_radius, theta0)
                    x1, y1 = _xy(centerline_radius, theta1)
                    z0 = contact_z0 + axial_offset
                    z1 = contact_z1 + axial_offset
                    direction = (x1 - x0, y1 - y0, z1 - z0)
                    height = math.sqrt(sum(value * value for value in direction))
                    self.capsule(
                        family="thread_rails_3",
                        path=section["rail_piece_path_template"].format(
                            start_index=start_index,
                            segment_index=segment_index,
                        ),
                        center=(
                            0.5 * (x0 + x1),
                            0.5 * (y0 + y1),
                            0.5 * (z0 + z1),
                        ),
                        radius=radius,
                        height=height,
                        direction=direction,
                    )
                    continue
                radii = (
                    inner_radius, outer_radius, outer_radius, inner_radius,
                    inner_radius, outer_radius, outer_radius, inner_radius,
                )
                thetas = (theta0, theta0, theta1, theta1) * 2
                zs = (
                    contact_z0, contact_z0, contact_z1, contact_z1,
                    contact_z0 + thickness, contact_z0 + thickness,
                    contact_z1 + thickness, contact_z1 + thickness,
                )
                points = []
                for radius, theta, z_value in zip(radii, thetas, zs):
                    x_value, y_value = _xy(radius, theta)
                    points.append(_point(x_value, y_value, z_value))
                self.mesh(
                    family="thread_rails_3",
                    path=section["rail_piece_path_template"].format(
                        start_index=start_index, segment_index=segment_index
                    ),
                    points=points,
                    counts=THREAD_COUNTS,
                    indices=THREAD_INDICES,
                    recipe_id="thread_rail_hexahedron_8v_v1",
                )
        for start_index, angle in enumerate(section["follower_local_center_angles_deg"]):
            self.mesh(
                family="thread_followers_3",
                path=section["follower_path_template"].format(start_index=start_index),
                points=_radial_tangent_box_points(
                    center_angle_deg=float(angle),
                    radial_interval=section["follower_radial_interval_m"],
                    tangential_width=float(section["follower_tangential_width_m"]),
                    axial_interval=section["follower_solid_interval_local_z_m"],
                ),
                counts=ANNULAR_COUNTS,
                indices=ANNULAR_INDICES,
                recipe_id="radial_tangent_box_8v_v1",
            )

    def _author_perforated_face(
        self,
        *,
        family: str,
        face: Mapping[str, Any],
        contacts: Sequence[tuple[str, float, float]],
        z_interval: Sequence[float],
        hole_vertex_radius: float,
    ) -> None:
        points_xy, triangles = _perforated_disk_triangulation(
            outer_radius=float(face["outer_radius_m"]),
            outer_segment_count=int(face["outer_polygon_segment_count"]),
            outer_phase_deg=float(face["outer_polygon_phase_deg"]),
            hole_centers=[(x_value, y_value) for _, x_value, y_value in contacts],
            hole_vertex_radius=float(hole_vertex_radius),
            hole_segment_count=int(face["hole_polygon_segment_count"]),
            hole_phase_deg=float(face["hole_polygon_phase_deg"]),
        )
        expected = int(face["expected_CDT_triangle_and_convex_prism_count"])
        if len(triangles) != expected:
            raise ValueError(f"{family} triangle count differs from frozen contract")
        for piece_index, triangle in enumerate(triangles):
            self.mesh(
                family=family,
                path=face["prism_path_template"].format(piece_index=piece_index),
                points=_triangle_prism_points(
                    [points_xy[index] for index in triangle],
                    float(z_interval[0]),
                    float(z_interval[1]),
                ),
                counts=TRI_PRISM_COUNTS,
                indices=TRI_PRISM_INDICES,
                recipe_id="planar_triangle_z_prism_6v_v1",
            )

    def author_electrical_contacts(self) -> None:
        section = self.blueprint["electrical_contacts"]
        nominal = self.document["public_geometry"]["contact_pattern_25_61"][
            "series_III_size20_interface_detail"
        ]["r7_collision_blueprint"]["deterministic_geometry_variants"]["nominal"][
            "authored_collider_values"
        ]
        fixed_contacts = self._contacts(plug_local=False)
        plug_contacts = self._contacts(plug_local=True)

        pins = section["pins"]
        pin_values = nominal["pins"]
        for label, x_value, y_value in fixed_contacts:
            self.cylinder(
                family="pins_61",
                path=pins["path_template"].format(label=label),
                center=(x_value, y_value, float(pin_values["center_local_z_m"])),
                radius=float(pins["radius_m"]),
                height=float(pin_values["height_m"]),
            )

        socket = section["socket_hard_entries"]
        socket_values = nominal["socket_entry"]
        for label, x_value, y_value in plug_contacts:
            for band in socket_values["profile_bands"]:
                band_index = int(band["profile_band_index"])
                for segment_index in range(24):
                    theta0 = segment_index * 15.0 - 7.5
                    theta1 = segment_index * 15.0 + 7.5
                    self.mesh(
                        family="hard_socket_entries_61",
                        path=(
                            f"{SUCCESSOR_ROOT_PRIM}/LoosePlug/BodyAssembly/Contacts/"
                            f"Socket_{label}/HardEntry/Band_{band_index:02d}/"
                            f"Wedge_{segment_index:02d}"
                        ),
                        points=_axial_profile_wedge_points(
                            center_xy=(x_value, y_value),
                            theta0_deg=theta0,
                            theta1_deg=theta1,
                            z0=float(band["depth_start_m"]),
                            z1=float(band["depth_end_m"]),
                            inner_z0=float(band["inner_radius_start_m"]),
                            outer_z0=float(band["outer_radius_m"]),
                            inner_z1=float(band["inner_radius_end_m"]),
                            outer_z1=float(band["outer_radius_m"]),
                        ),
                        counts=ANNULAR_COUNTS,
                        indices=ANNULAR_INDICES,
                        recipe_id="annular_wedge_8v_v1",
                    )

        petals = section["socket_petals"]
        for label, x_value, y_value in plug_contacts:
            for petal_index, phase in enumerate(petals["petal_phase_deg"]):
                points, counts, indices = _tangent_profile_prism(
                    center_xy=(x_value, y_value),
                    center_angle_deg=float(phase),
                    profile_depth_radius=petals["convex_depth_radial_profile_points_m"],
                    tangential_width=float(petals["tangential_width_m"]),
                )
                self.mesh(
                    family="socket_petals_366",
                    path=petals["path_template"].format(
                        label=label, petal_index=petal_index
                    ),
                    points=points,
                    counts=counts,
                    indices=indices,
                    recipe_id="tangent_profile_prism_v1",
                )

        hard_face = section["hard_insert_face"]
        hard_values = nominal["hard_insert_face"]
        self._author_perforated_face(
            family="hard_insert_face_prisms",
            face=hard_face,
            contacts=plug_contacts,
            z_interval=hard_values["local_depth_interval_m"],
            hole_vertex_radius=float(hard_values["hole_polygon_vertex_radius_m"]),
        )
        backing = section["fixed_backing_face"]
        backing_values = nominal["fixed_backing_face"]
        inherited_backing = dict(hard_face)
        inherited_backing.update(backing)
        inherited_backing["outer_polygon_phase_deg"] = hard_face["outer_polygon_phase_deg"]
        inherited_backing["hole_polygon_phase_deg"] = hard_face["hole_polygon_phase_deg"]
        self._author_perforated_face(
            family="fixed_backing_face_prisms",
            face=inherited_backing,
            contacts=fixed_contacts,
            z_interval=backing_values["local_z_interval_m"],
            hole_vertex_radius=float(backing_values["hole_polygon_vertex_radius_m"]),
        )

        barriers = self.blueprint["pin_barriers"]
        barrier_values = nominal["pin_barrier"]
        for label, x_value, y_value in fixed_contacts:
            for band in barrier_values["axial_profile_bands"]:
                band_index = int(band["profile_band_index"])
                for segment_index in range(int(barriers["angular_wedges_per_instance"])):
                    theta0 = segment_index * 15.0 - 7.5
                    theta1 = segment_index * 15.0 + 7.5
                    self.mesh(
                        family="pin_barriers_61",
                        path=(
                            f"{SUCCESSOR_ROOT_PRIM}/FixedReceptacle/Contacts/"
                            f"Barrier_{label}/Band_{band_index:02d}/"
                            f"Wedge_{segment_index:02d}"
                        ),
                        points=_axial_profile_wedge_points(
                            center_xy=(x_value, y_value),
                            theta0_deg=theta0,
                            theta1_deg=theta1,
                            z0=float(band["depth_start_m"]),
                            z1=float(band["depth_end_m"]),
                            inner_z0=float(band["inner_radius_start_m"]),
                            outer_z0=float(band["outer_radius_start_m"]),
                            inner_z1=float(band["inner_radius_end_m"]),
                            outer_z1=float(band["outer_radius_end_m"]),
                        ),
                        counts=ANNULAR_COUNTS,
                        indices=ANNULAR_INDICES,
                        recipe_id="annular_wedge_8v_v1",
                    )

    def author_force_and_stop_proxies(self) -> None:
        spring = self.blueprint["spring_fingers"]
        for index in range(int(spring["finger_count"])):
            local_angle = -8.0 - 30.0 * index
            points, counts, indices = _tangent_profile_prism(
                center_xy=(0.0, 0.0),
                center_angle_deg=local_angle,
                profile_depth_radius=spring["finger_convex_depth_radial_profile_points_m"],
                tangential_width=float(spring["finger_tangential_width_m"]),
            )
            self.mesh(
                family="spring_fingers_12",
                path=spring["finger_path_template"].format(segment_index=index),
                points=points,
                counts=counts,
                indices=indices,
                recipe_id="tangent_profile_prism_v1",
            )
            target_angle = 8.0 + 30.0 * index
            if spring["target_piece_shape"] == (
                "one_collision_isolated_analytic_cylinder"
            ):
                if index == 0:
                    self.cylinder(
                        family="receptacle_bore_targets_12",
                        path=str(spring["target_piece_path_template"]),
                        center=(
                            0.0,
                            0.0,
                            float(spring["target_cylinder_center_local_z_m"]),
                        ),
                        radius=float(spring["target_cylinder_radius_m"]),
                        height=float(spring["target_cylinder_height_m"]),
                    )
            elif spring["target_piece_shape"] == (
                "analytic_z_capsule_no_planar_entry_face"
            ):
                x_value, y_value = _xy(
                    float(spring["target_capsule_center_radius_m"]), target_angle
                )
                self.capsule(
                    family="receptacle_bore_targets_12",
                    path=spring["target_piece_path_template"].format(
                        segment_index=index
                    ),
                    center=(
                        x_value,
                        y_value,
                        float(spring["target_capsule_center_local_z_m"]),
                    ),
                    radius=float(spring["target_capsule_radius_m"]),
                    height=float(spring["target_capsule_cylinder_height_m"]),
                    direction=(0.0, 0.0, 1.0),
                    recipe_id=str(spring["target_primitive_recipe_id"]),
                )
            else:
                self.mesh(
                    family="receptacle_bore_targets_12",
                    path=spring["target_piece_path_template"].format(segment_index=index),
                    points=_radial_tangent_box_points(
                        center_angle_deg=target_angle,
                        radial_interval=(
                            float(spring["target_inner_bore_radius_m"]),
                            float(spring["target_outer_radius_m"]),
                        ),
                        tangential_width=float(spring["target_tangential_width_m"]),
                        axial_interval=spring["target_local_z_interval_m"],
                    ),
                    counts=ANNULAR_COUNTS,
                    indices=ANNULAR_INDICES,
                    recipe_id="radial_tangent_box_8v_v1",
                )

        seal = self.blueprint["peripheral_seal"]
        seal_step = float(seal["angular_step_deg"])
        for index in range(int(seal["segment_count"])):
            center = float(seal["phase_origin_deg"]) + index * seal_step
            common = dict(
                inner_radius=float(seal["radial_interval_m"][0]),
                outer_radius=float(seal["radial_interval_m"][1]),
                theta0_deg=center - 0.5 * seal_step,
                theta1_deg=center + 0.5 * seal_step,
            )
            self.mesh(
                family="seal_segments_24",
                path=seal["seal_path_template"].format(segment_index=index),
                points=_constant_wedge_points(
                    z0=float(seal["seal_local_z_interval_m"][0]),
                    z1=float(seal["seal_local_z_interval_m"][1]),
                    **common,
                ),
                counts=ANNULAR_COUNTS,
                indices=ANNULAR_INDICES,
                recipe_id="annular_wedge_8v_v1",
            )
            self.mesh(
                family="seal_targets_24",
                path=seal["target_path_template"].format(segment_index=index),
                points=_constant_wedge_points(
                    z0=float(seal["target_local_depth_interval_m"][0]),
                    z1=float(seal["target_local_depth_interval_m"][1]),
                    **common,
                ),
                counts=ANNULAR_COUNTS,
                indices=ANNULAR_INDICES,
                recipe_id="annular_wedge_8v_v1",
            )

        detent = self.blueprint["anti_decoupling_detent"]
        self.cylinder(
            family="detent_cam_continuous_base_1",
            path=str(detent["cam_base_path"]),
            center=(0.0, 0.0, float(detent["cam_local_center_z_m"])),
            radius=float(detent["cam_outer_base_radius_m"]),
            height=float(detent["cam_axial_width_m"]),
        )
        z0, z1 = (
            float(value) for value in detent["cam_tooth_local_z_interval_m"]
        )
        vertex_contract = detent["cam_tooth_CCW_vertex_order"]
        for tooth_index in range(int(detent["tooth_count"])):
            tooth_phase = float(detent["tooth0_phase_origin_deg"]) + tooth_index * float(
                detent["pitch_per_tooth_deg"]
            )
            polar_vertices = [
                (
                    tooth_phase
                    - float(vertex["positive_coupling_progress_deg"]),
                    float(vertex["radius_m"]),
                )
                for vertex in vertex_contract
            ]
            self.mesh(
                family="detent_cam_teeth_36",
                path=detent["cam_tooth_path_template"].format(
                    tooth_index=tooth_index
                ),
                points=_polar_triangle_prism_points(
                    polar_vertices=polar_vertices,
                    z0=z0,
                    z1=z1,
                ),
                counts=TRI_PRISM_COUNTS,
                indices=TRI_PRISM_INDICES,
                recipe_id="planar_triangle_z_prism_6v_v1",
            )

        if detent["follower_shape"] == "analytic_sphere":
            for index, angle in enumerate(detent["follower_phases_deg"]):
                x_value, y_value = _xy(
                    float(detent["follower_center_radius_m"]), float(angle)
                )
                self.sphere(
                    family="detent_followers_3",
                    path=detent["follower_path_template"].format(
                        follower_index=index
                    ),
                    center=(
                        x_value,
                        y_value,
                        float(detent["follower_local_center_z_m"]),
                    ),
                    radius=float(detent["follower_radius_m"]),
                )
        else:
            follower_z = (
                float(detent["follower_local_center_z_m"])
                - 0.5 * float(detent["follower_axial_width_m"]),
                float(detent["follower_local_center_z_m"])
                + 0.5 * float(detent["follower_axial_width_m"]),
            )
            for index, angle in enumerate(detent["follower_phases_deg"]):
                self.mesh(
                    family="detent_followers_3",
                    path=detent["follower_path_template"].format(follower_index=index),
                    points=_radial_tangent_box_points(
                        center_angle_deg=float(angle),
                        radial_interval=detent["follower_radial_interval_m"],
                        tangential_width=float(detent["follower_tangential_width_m"]),
                        axial_interval=follower_z,
                    ),
                    counts=ANNULAR_COUNTS,
                    indices=ANNULAR_INDICES,
                    recipe_id="radial_tangent_box_8v_v1",
                )

        bottom = self.blueprint["metal_bottoming"]
        if bottom.get("representation") == (
            "one_analytic_fixed_cap_plus_three_analytic_plug_spheres"
        ):
            self.cylinder(
                family="fixed_metal_stop_48",
                path=bottom["fixed_piece_path"],
                center=(0.0, 0.0, float(bottom["fixed_cap_center_local_z_m"])),
                radius=float(bottom["fixed_cap_radius_m"]),
                height=float(bottom["fixed_cap_axial_thickness_m"]),
            )
            for index, angle in enumerate(bottom["plug_sphere_phases_deg"]):
                x_value, y_value = _xy(
                    float(bottom["plug_sphere_distribution_radius_m"]),
                    float(angle),
                )
                self.sphere(
                    family="plug_metal_stop_48",
                    path=bottom["plug_piece_path_template"].format(
                        sphere_index=index
                    ),
                    center=(
                        x_value,
                        y_value,
                        float(bottom["plug_sphere_center_local_z_m"]),
                    ),
                    radius=float(bottom["plug_sphere_radius_m"]),
                )
        else:
            bottom_step = float(bottom["angular_step_deg"])
            for index in range(int(bottom["segment_count"])):
                center = float(bottom["phase_origin_deg"]) + index * bottom_step
                geometry = dict(
                    inner_radius=float(bottom["radial_interval_m"][0]),
                    outer_radius=float(bottom["radial_interval_m"][1]),
                    theta0_deg=center - 0.5 * bottom_step,
                    theta1_deg=center + 0.5 * bottom_step,
                )
                self.mesh(
                    family="fixed_metal_stop_48",
                    path=bottom["fixed_piece_path_template"].format(segment_index=index),
                    points=_constant_wedge_points(
                        z0=float(bottom["fixed_local_z_interval_m"][0]),
                        z1=float(bottom["fixed_local_z_interval_m"][1]),
                        **geometry,
                    ),
                    counts=ANNULAR_COUNTS,
                    indices=ANNULAR_INDICES,
                    recipe_id="annular_wedge_8v_v1",
                )
                self.mesh(
                    family="plug_metal_stop_48",
                    path=bottom["plug_piece_path_template"].format(segment_index=index),
                    points=_constant_wedge_points(
                        z0=float(bottom["plug_nominal_local_depth_interval_m"][0]),
                        z1=float(bottom["plug_nominal_local_depth_interval_m"][1]),
                        **geometry,
                    ),
                    counts=ANNULAR_COUNTS,
                    indices=ANNULAR_INDICES,
                    recipe_id="annular_wedge_8v_v1",
                )

        shoulder = self.blueprint["nut_body_shoulders"]
        shoulder_source = self.document["physical_proxy_boundaries"]["nut_body_bearing"][
            "physical_shoulder_collision_geometry"
        ]
        if shoulder["representation"] == (
            "two_analytic_axial_caps_each_against_three_analytic_spheres"
        ):
            groups = (
                ("positive", "shoulder_positive_body0_48", "shoulder_positive_body1_48"),
                ("negative", "shoulder_negative_body0_48", "shoulder_negative_body1_48"),
            )
            for sign, body0_family, body1_family in groups:
                stop = shoulder_source[f"{sign}_transZ_stop"]
                body0_path = self.families[body0_family]["path_templates"][0]
                self.cylinder(
                    family=body0_family,
                    path=body0_path,
                    center=(
                        0.0,
                        0.0,
                        float(stop["body0_collider_center_local_z_m"]),
                    ),
                    radius=float(shoulder_source["body0_cap_radius_m"]),
                    height=float(shoulder_source["body0_cap_axial_thickness_m"]),
                )
                for index, angle in enumerate(
                    shoulder_source["body1_sphere_phases_deg"]
                ):
                    x_value, y_value = _xy(
                        float(shoulder_source["body1_sphere_distribution_radius_m"]),
                        float(angle),
                    )
                    self.sphere(
                        family=body1_family,
                        path=self.families[body1_family]["path_templates"][0].format(
                            sphere_index=index
                        ),
                        center=(
                            x_value,
                            y_value,
                            float(stop["body1_sphere_center_local_z_m"]),
                        ),
                        radius=float(shoulder_source["body1_sphere_radius_m"]),
                    )
            return

        radial = (
            float(shoulder_source["radial_inner_m"]),
            float(shoulder_source["radial_outer_m"]),
        )
        thickness = float(shoulder_source["axial_thickness_m"])
        groups = (
            (
                "shoulder_positive_body0_48",
                f"{SUCCESSOR_ROOT_PRIM}{shoulder_source['positive_transZ_stop']['body0_collider_suffix']}",
                float(shoulder_source["positive_transZ_stop"]["body0_collider_center_local_z_m"]),
            ),
            (
                "shoulder_positive_body1_48",
                f"{SUCCESSOR_ROOT_PRIM}{shoulder_source['positive_transZ_stop']['body1_collider_suffix']}",
                float(shoulder_source["positive_transZ_stop"]["body1_collider_center_local_z_m"]),
            ),
            (
                "shoulder_negative_body0_48",
                f"{SUCCESSOR_ROOT_PRIM}{shoulder_source['negative_transZ_stop']['body0_collider_suffix']}",
                float(shoulder_source["negative_transZ_stop"]["body0_collider_center_local_z_m"]),
            ),
            (
                "shoulder_negative_body1_48",
                f"{SUCCESSOR_ROOT_PRIM}{shoulder_source['negative_transZ_stop']['body1_collider_suffix']}",
                float(shoulder_source["negative_transZ_stop"]["body1_collider_center_local_z_m"]),
            ),
        )
        shoulder_step = float(shoulder["angular_step_deg"])
        for family, group_path, center_z in groups:
            for index in range(int(shoulder["wedge_count_per_group"])):
                center = float(shoulder["phase_origin_deg"]) + index * shoulder_step
                self.mesh(
                    family=family,
                    path=f"{group_path}/Seg_{index:02d}",
                    points=_constant_wedge_points(
                        inner_radius=radial[0],
                        outer_radius=radial[1],
                        z0=center_z - 0.5 * thickness,
                        z1=center_z + 0.5 * thickness,
                        theta0_deg=center - 0.5 * shoulder_step,
                        theta1_deg=center + 0.5 * shoulder_step,
                    ),
                    counts=ANNULAR_COUNTS,
                    indices=ANNULAR_INDICES,
                    recipe_id="annular_wedge_8v_v1",
                )

    def author_collision_groups(self) -> None:
        expected_pairs, _ = _trusted_family_algebra(self.model)
        group_contract = self.blueprint["filtering"]["collision_group_authoring"]
        group_paths = {}
        for family in sorted(self.families):
            group_path = group_contract["group_path_template"].format(
                primitive_family=family
            )
            _ensure_xform(self.stage, str(Path(group_path).parent), self.UsdGeom)
            group = self.UsdPhysics.CollisionGroup.Define(self.stage, group_path)
            group.CreateInvertFilteredGroupsAttr(False)
            collection = group.GetCollidersCollectionAPI()
            collection.CreateExpansionRuleAttr("explicitOnly")
            collection.CreateIncludeRootAttr(False)
            collection.CreateIncludesRel().SetTargets(
                [self.Sdf.Path(path) for path in sorted(self.family_paths[family])]
            )
            _custom_string(group.GetPrim(), "kcg:primitiveFamily", family, self.Sdf)
            group_paths[family] = group_path
        for (left, right), row in sorted(expected_pairs.items()):
            if row["expected_decision"] != "filtered":
                continue
            source = self.UsdPhysics.CollisionGroup.Get(
                self.stage, group_paths[left]
            )
            source.CreateFilteredGroupsRel().AddTarget(self.Sdf.Path(group_paths[right]))

    def verify_inventory(self) -> None:
        expected = _trusted_collider_inventory(self.model)
        expected_by_family: dict[str, set[str]] = defaultdict(set)
        for family, path, _ in expected:
            expected_by_family[family].add(path)
        if set(self.family_paths) != set(expected_by_family):
            raise ValueError("authored primitive-family inventory differs")
        for family in sorted(expected_by_family):
            actual = set(self.family_paths[family])
            if actual != expected_by_family[family]:
                raise ValueError(
                    f"{family} path inventory differs: "
                    f"missing={len(expected_by_family[family] - actual)}, "
                    f"unexpected={len(actual - expected_by_family[family])}"
                )
        if len(self.authored_colliders) != len(expected):
            raise ValueError("total authored collider count differs")
        root = self.stage.GetPrimAtPath(SUCCESSOR_ROOT_PRIM)
        root.CreateAttribute(
            "kcg:authoredColliderCount", self.Sdf.ValueTypeNames.Int, custom=True
        ).Set(len(self.authored_colliders))
        root.CreateAttribute(
            "kcg:primitiveFamilyCount", self.Sdf.ValueTypeNames.Int, custom=True
        ).Set(len(self.family_paths))

    def author_all(self) -> None:
        self.author_identity_and_rigid_bodies()
        self.author_joint()
        self.author_shells_and_keying()
        self.author_thread()
        self.author_electrical_contacts()
        self.author_force_and_stop_proxies()
        self.author_collision_groups()
        self.verify_inventory()


def _authorized_output(model: Any, requested: str | None) -> Path:
    identity = model.document["identity"]
    authorized = (
        WORKSPACE_ROOT
        / str(identity["recommended_asset_directory"])
        / str(identity["recommended_asset_name"])
    ).resolve()
    output = authorized if requested is None else Path(requested).expanduser().resolve()
    if output != authorized:
        raise ValueError("nominal asset may only be authored at its contract output path")
    if identity["overwrite_existing"] is not False:
        raise ValueError("successor asset overwrite guard changed")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing successor asset: {output}")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    config_path = Path(arguments.config).expanduser().resolve()
    if config_path.name == "d38999_keyed_v3_physical_model_contract_r12_v1.yaml":
        from kcg_connector.d38999_keyed_v3_physical_r12_contract import (
            candidate_model,
            load_r12_physical_model_contract,
        )

        model = load_r12_physical_model_contract(config_path)
        if arguments.candidate_index is not None:
            model = candidate_model(model, arguments.candidate_index)
    else:
        if arguments.candidate_index is not None:
            raise ValueError("candidate index is available only for the r12 contract")
        model = load_physical_model_contract(config_path)
    if not model.a2_asset_authoring_allowed:
        raise PermissionError("A2 asset authoring remains blocked by A0")
    if arguments.geometry_variant != "nominal":
        raise ValueError("the authorized A2 successor identity is the nominal variant")
    output = _authorized_output(model, arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".authoring_tmp" + output.suffix)
    if temporary.exists():
        raise FileExistsError(f"stale authoring temporary exists: {temporary}")

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.CreateNew(str(temporary))
    if stage is None:
        raise RuntimeError("could not create the temporary USD stage")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    authorer = _AssetAuthorer(
        stage=stage,
        model=model,
        Gf=Gf,
        Sdf=Sdf,
        UsdGeom=UsdGeom,
        UsdPhysics=UsdPhysics,
        UsdShade=UsdShade,
    )
    authorer.author_all()
    stage.GetRootLayer().Save()
    stage = None
    temporary.replace(output)

    reopened = Usd.Stage.Open(str(output), load=Usd.Stage.LoadAll)
    if reopened is None or not reopened.GetPrimAtPath(SUCCESSOR_ROOT_PRIM).IsValid():
        raise RuntimeError("saved successor asset cannot be reopened at its frozen root")
    print(f"created={output}")
    print(f"colliders={len(authorer.authored_colliders)}")
    print(f"primitive_families={len(authorer.family_paths)}")
    print("geometry_variant=nominal")
    print("downstream_authorized=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
