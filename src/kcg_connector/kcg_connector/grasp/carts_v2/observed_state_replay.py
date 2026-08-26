"""Offline exact-mesh replay of observed nail-free hand states; never online."""

from __future__ import annotations

from collections.abc import Mapping
import itertools

import numpy as np

try:
    import fcl
except ImportError:  # Fail closed when the registered mesh backend is absent.
    fcl = None

from kcg_connector.grasp.carts_v2.fast_filter import build_fcl_bvh_model
from kcg_connector.grasp.carts_v2.height_projection import minimum_z_over_finite_table_top
from kcg_connector.grasp.carts_v2.surface_contact import ExactContactSurfaceQuery
from kcg_connector.grasp.carts_v2.surface_contact import nearest_motion_compatible_index
from kcg_connector.grasp.carts_v2.task_grip_surface import allowed_object_grasp_center_m
from kcg_connector.grasp.carts_v2.task_grip_surface import motion_compatible_with_object_witness
from kcg_connector.grasp.carts_v2.task_grip_surface import task_noncontact_triangles
from kcg_connector.robot_model import ALL_HAND_JOINT_NAMES, MIMIC_HAND_JOINTS


def _rigid_transform(value, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if (
        matrix.shape != (4, 4)
        or not np.all(np.isfinite(matrix))
        or not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-12)
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1.0e-9)
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1.0, atol=1.0e-9)
    ):
        raise ValueError(f"{label} must be one finite rigid 4x4 transform")
    return matrix


def _fcl_transform(matrix: np.ndarray):
    return fcl.Transform(matrix[:3, :3], matrix[:3, 3])


def _distance_and_intersection(left, right) -> tuple[float, bool]:
    collision = bool(fcl.collide(
        left, right, fcl.CollisionRequest(num_max_contacts=1, enable_contact=False),
        fcl.CollisionResult(),
    ))
    if collision:
        return 0.0, True
    distance = float(fcl.distance(
        left, right, fcl.DistanceRequest(enable_nearest_points=False),
        fcl.DistanceResult(),
    ))
    if not np.isfinite(distance) or distance < 0.0:
        raise RuntimeError("FCL returned a nonfinite free-space mesh distance")
    return distance, False


class ObservedHandStateEvaluator:
    """Cache exact geometry and replay actual hand/object states offline only."""

    def __init__(self, inputs) -> None:
        if fcl is None:
            raise RuntimeError("python-fcl mesh backend is unavailable")
        if inputs.task_grip_surfaces is None:
            raise ValueError("observed replay requires registered TASK_GRIP_SURFACE")
        self.inputs = inputs
        registered = inputs.hand_collision_triangles_by_link
        non_task = task_noncontact_triangles(registered, inputs.task_grip_surfaces)
        self._full_hand = {
            name: fcl.CollisionObject(build_fcl_bvh_model(triangles))
            for name, triangles in registered.items()
        }
        self._non_task_hand = {
            name: fcl.CollisionObject(build_fcl_bvh_model(triangles))
            for name, triangles in non_task.items()
        }
        links = tuple(sorted(self._full_hand))
        adjacent = {
            tuple(sorted((joint.parent_link, joint.child_link)))
            for joint in inputs.hand_model.joints.values()
            if joint.parent_link in self._full_hand and joint.child_link in self._full_hand
        }
        self._self_pairs = tuple(
            pair for pair in itertools.combinations(links, 2) if pair not in adjacent
        )
        if len(links) != 9 or len(adjacent) != 8 or len(self._self_pairs) != 28:
            raise ValueError("registered hand self-collision pair coverage changed")
        mesh = inputs.object_contract.model.mesh
        self._object = fcl.CollisionObject(
            build_fcl_bvh_model(mesh.vertices_m, mesh.faces)
        )
        self._surface_query = ExactContactSurfaceQuery(inputs)
        self._object_center = allowed_object_grasp_center_m(inputs)
        self._contact_distance = float(
            inputs.config.section("closure_prediction")["contact_distance_m"]
        )
        self._minimum_motion = float(inputs.config.section(
            "closure_prediction")["minimum_inward_motion_m_per_phase"])
        self._maximum_witnesses = int(inputs.config.section(
            "closure_prediction")["nearest_face_candidate_count"])
        self._table_tolerance = float(inputs.config.section(
            "fast_filter")["table_penetration_tolerance_m"])

    def _state(self, world_from_handbase, joint_positions_by_name, world_from_object):
        hand = _rigid_transform(world_from_handbase, "world_from_handbase")
        obj = _rigid_transform(world_from_object, "world_from_object")
        supplied = {str(name): float(value)
                    for name, value in dict(joint_positions_by_name).items()}
        if set(supplied) != set(ALL_HAND_JOINT_NAMES):
            raise ValueError("observed state must contain exactly eight named hand joints")
        model = self.inputs.hand_model
        independent = {name: supplied[name] for name in model.independent_joint_names}
        resolved = model.resolve_joint_positions(independent, enforce_limits=True)
        for name, value in supplied.items():
            joint = model.joints[name]
            if not np.isfinite(value) or (joint.limit is not None
                    and not joint.limit.contains(value)):
                raise ValueError(f"observed joint {name} is nonfinite or outside limits")

        def actual_fk(base):
            transforms = {model.base_link: np.asarray(base, np.float64).copy()}
            for name in model.joint_order:
                joint = model.joints[name]
                parent = transforms[joint.parent_link]
                position = supplied[name] if joint.movable else 0.0
                transforms[joint.child_link] = (parent @ joint.origin_transform()
                    @ joint.motion_transform(position))
            return transforms

        links_world = actual_fk(hand)
        object_from_hand = np.linalg.inv(obj) @ hand
        links_object = actual_fk(object_from_hand)
        return hand, obj, supplied, resolved, links_world, links_object

    def _table(self, links_world) -> dict[str, object]:
        best = None
        for link, triangles in self.inputs.hand_collision_triangles_by_link.items():
            transform = links_world[link]
            world = triangles @ transform[:3, :3].T + transform[:3, 3]
            minimum_z, triangle = minimum_z_over_finite_table_top(
                world, self.inputs.table_xy_bounds_m)
            if minimum_z is None:
                continue
            row = (float(minimum_z - self.inputs.table_top_z_m), link, int(triangle))
            if best is None or row < best:
                best = row
        clearance, link, triangle = (None, "", None) if best is None else best
        intersects = bool(
            clearance is not None and clearance < -self._table_tolerance)
        return {
            "minimum_clearance_m": clearance,
            "minimum_clearance_link": link,
            "minimum_clearance_triangle_index": triangle,
            "top_intersection_beyond_numerical_tolerance": intersects,
            "evidence_scope": "FINITE_XY_TABLE_TOP_VERTICAL_CLEARANCE_NOT_SIDE_OR_BOTTOM",
        }

    def _self_collision(self, links_world) -> dict[str, object]:
        for link, collision_object in self._full_hand.items():
            collision_object.setTransform(_fcl_transform(links_world[link]))
        collisions, best = [], None
        for pair in self._self_pairs:
            distance, intersects = _distance_and_intersection(
                self._full_hand[pair[0]], self._full_hand[pair[1]])
            if intersects:
                collisions.append(pair)
            row = (distance, pair)
            if best is None or row < best:
                best = row
        return {
            "intersecting_pairs": [list(pair) for pair in collisions],
            "minimum_clearance_m": best[0],
            "minimum_clearance_pair": list(best[1]),
            "checked_nonadjacent_pair_count": len(self._self_pairs),
            "direct_parent_child_pairs_excluded": True,
        }

    def _non_task_object(self, links_world, world_from_object) -> dict[str, object]:
        self._object.setTransform(_fcl_transform(world_from_object))
        collisions, best = [], None
        for link, collision_object in sorted(self._non_task_hand.items()):
            collision_object.setTransform(_fcl_transform(links_world[link]))
            distance, intersects = _distance_and_intersection(collision_object, self._object)
            if intersects:
                collisions.append(link)
            row = (distance, link)
            if best is None or row < best:
                best = row
        return {
            "intersecting_links": collisions,
            "minimum_clearance_m": best[0],
            "minimum_clearance_link": best[1],
            "allowed_task_faces_subtracted_from_terminal_links": True,
        }

    def _phase_delta(self, pad_name, previous_resolved, current_resolved) -> float:
        pad_names = tuple(pad.name for pad in self.inputs.hand_contract.pads)
        row = self.inputs.closing_directions[pad_names.index(pad_name)]
        names = self.inputs.hand_model.independent_joint_names
        lower, upper = self.inputs.hand_model.joint_limit_vectors()
        values = []
        for index in np.flatnonzero(row):
            delta = current_resolved[names[index]] - previous_resolved[names[index]]
            values.append(float(np.sign(row[index]) * delta / (upper[index] - lower[index])))
        return 0.0 if not values else float(np.mean(values))

    def _task_contacts(self, current, previous) -> tuple[dict[str, object], bool]:
        rows, ambiguous = {}, False
        links_object = current[5]
        for pad_name, surface in sorted(self.inputs.task_grip_surfaces.items()):
            transform = links_object[surface.link_name]
            nearest, pad_points, object_normals = (
                self._surface_query.query_task_surface_witnesses(
                    pad_name, transform, self._maximum_witnesses))
            selected = int(np.argmin(nearest.distance_m))
            allowed_distance = float(nearest.distance_m[selected])
            forbidden = nearest.forbidden_distance_m
            tolerance = 64.0 * np.finfo(np.float64).eps
            forbidden_first = bool(
                forbidden is not None and forbidden <= self._contact_distance + tolerance
                and forbidden <= allowed_distance + tolerance)
            contact = bool(
                allowed_distance <= self._contact_distance + tolerance
                and not forbidden_first and not nearest.intersecting)
            first_contact = motion_compatible = None
            phase_delta = None
            if previous is not None:
                previous_link = previous[5][surface.link_name]
                previous_nearest, _, _ = self._surface_query.query_task_surface_witnesses(
                    pad_name, previous_link, self._maximum_witnesses)
                previous_distance = float(np.min(previous_nearest.distance_m))
                previous_forbidden = previous_nearest.forbidden_distance_m
                previous_forbidden_first = bool(
                    previous_forbidden is not None
                    and previous_forbidden <= self._contact_distance + tolerance
                    and previous_forbidden <= previous_distance + tolerance)
                previous_contact = bool(
                    previous_distance <= self._contact_distance + tolerance
                    and not previous_nearest.intersecting
                    and not previous_forbidden_first)
                first_contact = contact and not previous_contact
                if first_contact:
                    phase_delta = self._phase_delta(
                        pad_name, previous[3], current[3])
                    if phase_delta > 64.0 * np.finfo(np.float64).eps:
                        local = ((pad_points - transform[:3, 3])
                                 @ transform[:3, :3])
                        current_object_from_previous_link = (
                            np.linalg.inv(current[1]) @ previous[4][surface.link_name])
                        previous_points = (
                            local @ current_object_from_previous_link[:3, :3].T
                            + current_object_from_previous_link[:3, 3])
                        motion = (pad_points - previous_points) / phase_delta
                        assert nearest.surface_normal_m is not None
                        compatible = motion_compatible_with_object_witness(
                            pad_points, nearest.point_m, nearest.surface_normal_m,
                            object_normals, motion, self._object_center,
                            self._minimum_motion)
                        inward = -np.einsum("ij,ij->i", motion, object_normals)
                        motion_index = nearest_motion_compatible_index(
                            nearest.distance_m, np.where(compatible, inward, -np.inf),
                            self._minimum_motion)
                        motion_compatible = bool(
                            motion_index >= 0
                            and nearest.distance_m[motion_index]
                            <= self._contact_distance + tolerance)
                    else:
                        motion_compatible = False
            rows[pad_name] = {
                "task_grip_surface_contact": contact,
                "allowed_distance_m": allowed_distance,
                "object_allowed_face_index": int(nearest.face_index[selected]),
                "hand_source_face_index": int(nearest.surface_face_index[selected]),
                "legacy_blue_pad": bool(nearest.surface_legacy_blue_pad[selected]),
                "forbidden_distance_m": forbidden,
                "forbidden_object_face_index": nearest.forbidden_face_index,
                "forbidden_first": forbidden_first,
                "full_object_intersecting": bool(nearest.intersecting),
                "registered_patch_count": nearest.registered_patch_count,
                "finite_patch_witness_count": nearest.finite_patch_witness_count,
                "first_contact_from_previous_state": first_contact,
                "first_contact_phase_delta": phase_delta,
                "first_contact_motion_compatible": motion_compatible,
            }
            ambiguous = ambiguous or bool(nearest.intersecting)
        return rows, ambiguous

    def evaluate(
        self,
        world_from_handbase,
        joint_positions_by_name: Mapping[str, float],
        world_from_object,
        *,
        previous_state: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Replay one saved state; never call this result from online control."""
        current = self._state(
            world_from_handbase, joint_positions_by_name, world_from_object)
        previous = None
        if previous_state is not None:
            previous = self._state(
                previous_state["world_from_handbase"],
                previous_state["joint_positions_by_name"],
                previous_state["world_from_object"],
            )
        table = self._table(current[4])
        self_collision = self._self_collision(current[4])
        non_task = self._non_task_object(current[4], current[1])
        task_contacts, ambiguous = self._task_contacts(current, previous)
        fail_closed = bool(
            table["top_intersection_beyond_numerical_tolerance"]
            or self_collision["intersecting_pairs"]
            or non_task["intersecting_links"]
            or ambiguous)
        mimic_errors = {
            follower: float(current[2][follower] - current[3][follower])
            for follower in MIMIC_HAND_JOINTS
        }
        return {
            "status": ("OBSERVED_STATE_GEOMETRY_REJECT" if fail_closed
                       else "OBSERVED_STATE_GEOMETRY_REPLAYED"),
            "offline_post_run_only": True,
            "online_control_use_allowed": False,
            "hand_variant": self.inputs.hand_variant,
            "mimic_error_rad_by_joint": mimic_errors,
            "table_top": table,
            "self_collision": self_collision,
            "non_task_hand_object": non_task,
            "task_grip_surface_by_finger": task_contacts,
            "fail_closed": fail_closed,
            "fail_closed_reason": (
                "MESH_INTERSECTION_OR_UNCLASSIFIED_TASK_SURFACE_INTERSECTION"
                if fail_closed else ""),
        }


__all__ = ["ObservedHandStateEvaluator"]
