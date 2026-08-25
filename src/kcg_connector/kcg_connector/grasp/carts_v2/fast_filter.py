"""Cheap hard rejection for all V2 candidates; never a safety certificate."""
from __future__ import annotations
import itertools
import json
import math

import numpy as np

try:
    import fcl
except ImportError:  # The fast gate must reject, never silently skip.
    fcl = None

from kcg_connector.grasp.carts_v2.models import (
    CandidateSeed, ClosurePrediction, FastFilterResult, V2Inputs,
    joint_positions_for_phases,
)
from kcg_connector.grasp.carts_v2.height_projection import (
    minimum_z_over_finite_table_top,
)
from kcg_connector.grasp.robust.object_model import load_stl_mesh
from kcg_connector.grasp.carts_v2.task_grip_surface import task_noncontact_triangles

_NONPAD_POLICY = "FCL_EXACT_NONPAD_AND_SELF_CONTROL_STATES_SAMPLED"
def _exact_nonpad_surfaces(inputs: V2Inputs) -> dict[str, np.ndarray]:
    if inputs.task_grip_surfaces is not None:
        return task_noncontact_triangles(inputs.hand_collision_triangles_by_link,
                                         inputs.task_grip_surfaces)
    manifest_path = inputs.hand_contract.source_manifest.absolute_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("links", ())
    pads = {pad.link_name: pad for pad in inputs.hand_contract.pads}
    by_link = {str(row.get("link_name", "")): row for row in rows}
    if (
        manifest.get("schema") != "CARTS_EXACT_SOURCE_TERMINAL_PAD_V1"
        or manifest.get("coordinate_tolerance_used") is not False
        or len(rows) != 3
        or set(by_link) != set(pads)
    ):
        raise ValueError("terminal PAD exact-source manifest changed")
    result = dict(inputs.hand_collision_triangles_by_link)
    for link_name, pad in pads.items():
        row = by_link[link_name]
        source = (inputs.repository_root / str(row["source_mesh"])).resolve()
        if inputs.repository_root not in source.parents:
            raise ValueError("terminal source mesh resolves outside repository")
        mesh, provenance = load_stl_mesh(source, unit="m", orient_outward=False)
        arrays = (manifest_path.parent / str(row["pad_source_arrays"])).resolve()
        diagnostics = row.get("diagnostics", {})
        if (
            provenance.source_sha256 != row.get("source_mesh_sha256")
            or len(mesh.faces) != int(diagnostics.get("source_face_count", -1))
            or diagnostics.get("exact_source_face_ordinal_lineage_complete") is not True
            or arrays != pad.mesh.absolute_path
            or row.get("pad_source_arrays_sha256") != pad.mesh.sha256
        ):
            raise ValueError(f"terminal PAD lineage changed for {link_name}")
        with np.load(arrays, allow_pickle=False) as archive:
            indices = np.asarray(archive["source_face_indices"])
        if (
            indices.ndim != 1
            or not np.issubdtype(indices.dtype, np.integer)
            or len(indices) != pad.triangle_count
            or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0)
            or np.any(indices >= len(mesh.faces))
        ):
            raise ValueError(f"terminal PAD source ordinals invalid for {link_name}")
        indices = indices.astype(np.int64, copy=False)
        if not np.array_equal(mesh.face_vertices_m[indices], pad.points_local_m[pad.faces]):
            raise ValueError(f"terminal PAD source triangles changed for {link_name}")
        keep = np.ones(len(mesh.faces), dtype=np.bool_)
        keep[indices] = False
        result[link_name] = np.asarray(mesh.face_vertices_m[keep], dtype=np.float64)
    return result
def build_fcl_bvh_model(vertices: np.ndarray, faces: np.ndarray | None = None):
    if fcl is None:
        raise RuntimeError("python-fcl mesh backend is unavailable")
    if faces is None:
        vertices = np.asarray(vertices).reshape(-1, 3)
        faces = np.arange(len(vertices), dtype=np.int32).reshape(-1, 3)
    model = fcl.BVHModel()
    model.beginModel(len(vertices), len(faces))
    model.addSubModel(np.ascontiguousarray(vertices, dtype=np.float64),
                      np.ascontiguousarray(faces, dtype=np.int32))
    model.endModel()
    return model
def _prepare_fcl_scene(inputs: V2Inputs):
    if fcl is None:
        return None
    registered = inputs.hand_collision_triangles_by_link
    nonpad = _exact_nonpad_surfaces(inputs)
    self_objects = {name: fcl.CollisionObject(build_fcl_bvh_model(triangles))
                    for name, triangles in registered.items()}
    nonpad_objects = {name: fcl.CollisionObject(build_fcl_bvh_model(triangles))
                      for name, triangles in nonpad.items()}
    contact_geometry = ({pad.name: (pad.link_name, pad.points_local_m, pad.faces)
                         for pad in inputs.hand_contract.pads}
                        if inputs.task_grip_surfaces is None else
                        {name: (row.link_name, row.points_local_m, row.faces)
                         for name, row in inputs.task_grip_surfaces.items()})
    contact_objects = {name: (link, fcl.CollisionObject(build_fcl_bvh_model(points, faces)))
        for name, (link, points, faces) in contact_geometry.items()}
    links = tuple(sorted(self_objects))
    adjacent = {
        tuple(sorted((joint.parent_link, joint.child_link)))
        for joint in inputs.hand_model.joints.values()
        if joint.parent_link in self_objects and joint.child_link in self_objects
    }
    pairs = tuple(pair for pair in itertools.combinations(links, 2)
                  if pair not in adjacent)
    if len(links) != 9 or len(adjacent) != 8 or len(pairs) != 28:
        raise ValueError("registered hand self-collision pair coverage changed")
    mesh = inputs.object_contract.model.mesh
    world_from_object = inputs.frozen_world_from_object
    object_collision = fcl.CollisionObject(build_fcl_bvh_model(mesh.vertices_m, mesh.faces),
        fcl.Transform(world_from_object[:3, :3], world_from_object[:3, 3]))
    return self_objects, nonpad_objects, object_collision, pairs, contact_objects
def _first_state_collision(inputs: V2Inputs, states, scene) -> str:
    if scene is None:
        return "FCL_MESH_COLLISION_BACKEND_UNAVAILABLE"
    self_objects, nonpad_objects, object_collision, pairs, contact_objects = scene
    request = fcl.CollisionRequest(num_max_contacts=1, enable_contact=False)
    for stage, base, joints in states:
        transforms = inputs.hand_model.forward_kinematics(joints, base_transform=base)
        for link_name in self_objects:
            transform = transforms[link_name]
            fcl_transform = fcl.Transform(transform[:3, :3], transform[:3, 3])
            self_objects[link_name].setTransform(fcl_transform)
            nonpad_objects[link_name].setTransform(fcl_transform)
        for _name, (link_name, collision_object) in contact_objects.items():
            transform = transforms[link_name]
            collision_object.setTransform(fcl.Transform(transform[:3, :3],
                                                        transform[:3, 3]))
        for first, second in pairs:
            if fcl.collide(self_objects[first], self_objects[second], request,
                           fcl.CollisionResult()):
                return f"NONADJACENT_HAND_SELF_COLLISION:{stage}:{first}:{second}"
        for link_name in sorted(nonpad_objects):
            if fcl.collide(nonpad_objects[link_name], object_collision, request,
                           fcl.CollisionResult()):
                return f"NONPAD_HAND_OBJECT_COLLISION:{stage}:{link_name}"
        task_items = (() if inputs.task_grip_surfaces is None
                      else sorted(contact_objects.items()))
        for name, (_link, collision_object) in task_items:
            if fcl.collide(collision_object, object_collision, request,
                           fcl.CollisionResult()):
                return f"TASK_GRIP_SURFACE_OBJECT_INTERSECTION:{stage}:{name}"
    return ""
def _hard_reasons(inputs: V2Inputs, prediction: ClosurePrediction) -> list[str]:
    if prediction.status != "CLOSURE_SURVIVE":
        return [prediction.reason or "CLOSURE_REJECT"]
    reasons: list[str] = []
    contacts = prediction.contacts
    expected_pads = {pad.name for pad in inputs.hand_contract.pads}
    if len(contacts) != 3 or {contact.pad_name for contact in contacts} != expected_pads:
        reasons.append("THREE_DISTINCT_REGISTERED_PADS_NOT_PRESENT")
    face_count = len(inputs.object_contract.model.mesh.faces)
    if any(not 0 <= contact.object_face_index < face_count for contact in contacts):
        reasons.append("CONTACT_FACE_INDEX_OUT_OF_RANGE")
    elif any(
        not inputs.face_roles.face_is_allowed[contact.object_face_index]
        for contact in contacts
    ):
        reasons.append("CONTACT_ON_FORBIDDEN_FACE")
    minimum_area = float(
        inputs.config.section("fast_filter")[
            "minimum_three_contact_triangle_area_m2"
        ]
    )
    areas = inputs.object_contract.model.mesh.face_areas_m2
    if any(areas[contact.object_face_index] < minimum_area for contact in contacts):
        reasons.append("CONTACT_TRIANGLE_TOO_SMALL_FOR_FAST_MODEL")
    try:
        inputs.hand_model.resolve_joint_positions(
            prediction.final_joint_positions_rad, enforce_limits=True
        )
    except ValueError:
        reasons.append("JOINT_LIMIT_VIOLATION")
    if not np.all(np.isfinite(prediction.seed.object_from_hand_matrix())):
        reasons.append("NONFINITE_PALM_POSE")
    return reasons
def _approach_states(
    inputs: V2Inputs, seed: CandidateSeed, joints: np.ndarray
) -> tuple[tuple[str, np.ndarray, np.ndarray], ...]:
    settings = inputs.config.section("fast_filter")
    dynamic = inputs.config.section("dynamic")
    base = inputs.frozen_world_from_object @ seed.object_from_hand_matrix()
    height = float(dynamic["approach_clearance_height_m"])
    sample_count = int(settings["approach_path_sample_count"])
    direction_object = seed.approach_direction_object
    if direction_object is None:
        direction_world = np.asarray((0.0, 0.0, -1.0), dtype=np.float64)
    else:
        direction_object_array = np.asarray(direction_object, dtype=np.float64)
        if direction_object_array.shape != (3,):
            raise ValueError("approach direction must be one finite unit vector")
        direction_world = (
            inputs.frozen_world_from_object[:3, :3] @ direction_object_array
        )
    direction_norm = float(np.linalg.norm(direction_world))
    if (
        direction_world.shape != (3,)
        or not np.isfinite(direction_norm)
        or abs(direction_norm - 1.0) > 1.0e-6
    ):
        raise ValueError("approach direction must be one finite unit vector")
    states = []
    for index, fraction in enumerate(np.linspace(1.0, 0.0, sample_count)):
        shifted = np.array(base, copy=True)
        shifted[:3, 3] -= direction_world * height * float(fraction)
        stage = "PREGRASP" if index == sample_count - 1 else f"APPROACH_{index:02d}"
        states.append((stage, shifted, joints))
    return tuple(states)
def sampled_sequential_closure_states(inputs: V2Inputs, seed: CandidateSeed,
    final_phases: tuple[float, float, float]
) -> tuple[tuple[str, np.ndarray, np.ndarray], ...]:
    dynamic = inputs.config.section("dynamic")
    pregrasp = np.asarray(seed.pregrasp_joint_positions_rad)
    approach = _approach_states(inputs, seed, pregrasp)
    base = approach[-1][1]
    states = list(approach)
    phases = list(seed.pregrasp_closure_phases)
    phase_by_pad = {
        pad.name: index for index, pad in enumerate(inputs.hand_contract.pads)
    }
    maximum_increment = _control_increment(inputs)
    for stop_index, pad_name in enumerate(
        inputs.config.section("closure_prediction")["closing_order"], start=1
    ):
        phase_index = phase_by_pad[str(pad_name)]
        start_phases = tuple(phases)
        phases[phase_index] = final_phases[phase_index]
        stop_phases = tuple(phases)
        start_joints = joint_positions_for_phases(
            inputs, start_phases, reference_joint_positions_rad=pregrasp
        )
        stop_joints = joint_positions_for_phases(
            inputs, stop_phases, reference_joint_positions_rad=pregrasp
        )
        largest_change = float(np.max(np.abs(stop_joints - start_joints)))
        step_count = max(1, int(math.ceil(largest_change / maximum_increment)))
        for step_index in range(1, step_count + 1):
            fraction = step_index / step_count
            sample_phases = list(start_phases)
            sample_phases[phase_index] += fraction * (
                stop_phases[phase_index] - start_phases[phase_index]
            )
            stage = (
                f"CONTACT_STOP_{stop_index}"
                if step_index == step_count
                else f"FINGER_{stop_index}_CLOSURE_{step_index:04d}"
            )
            joints = joint_positions_for_phases(
                inputs, tuple(sample_phases), reference_joint_positions_rad=pregrasp
            )
            states.append((stage, base, joints))
    return tuple(states)


def _state_table_clearance(
    inputs: V2Inputs, base: np.ndarray, joints: np.ndarray
) -> tuple[float | None, str]:
    transforms = inputs.hand_model.forward_kinematics(
        joints, base_transform=base
    )
    minimum: tuple[float, str] | None = None
    for link_name, triangles in inputs.hand_collision_triangles_by_link.items():
        transform = transforms[link_name]
        world = triangles @ transform[:3, :3].T + transform[:3, 3]
        minimum_z, _triangle_index = minimum_z_over_finite_table_top(
            world, inputs.table_xy_bounds_m)
        if minimum_z is None:
            continue
        gap = float(minimum_z - inputs.table_top_z_m)
        if minimum is None or gap < minimum[0]:
            minimum = (gap, link_name)
    return (None, "") if minimum is None else minimum
def _state_clearance_summary(
    inputs: V2Inputs,
    states: tuple[tuple[str, np.ndarray, np.ndarray], ...],
    *,
    violation_below_m: float | None = None,
) -> tuple[
    tuple[float, str, str, tuple[float, ...]] | None,
    tuple[float, str, str] | None,
]:
    minimum: tuple[float, str, str, tuple[float, ...]] | None = None
    first_violation: tuple[float, str, str] | None = None
    for stage, base, joints in states:
        gap, link_name = _state_table_clearance(inputs, base, joints)
        if gap is not None and (minimum is None or gap < minimum[0]):
            minimum = (gap, link_name, stage,
                       tuple(float(value) for value in joints))
        if (
            first_violation is None
            and gap is not None
            and violation_below_m is not None
            and gap < violation_below_m
        ):
            first_violation = (gap, link_name, stage)
    return minimum, first_violation
def _maximum_joint_increment(
    states: tuple[tuple[str, np.ndarray, np.ndarray], ...],
) -> float:
    return max((float(np.max(np.abs(right[2] - left[2])))
                for left, right in zip(states, states[1:])), default=0.0)
def _control_increment(inputs: V2Inputs) -> float:
    dynamic = inputs.config.section("dynamic")
    return float(dynamic["finger_maximum_speed_rad_s"]) * float(dynamic["physics_dt_s"])
def sampled_pregrasp_path_states(
    inputs: V2Inputs, seed: CandidateSeed, phases: tuple[float, float, float]
) -> tuple[tuple[str, np.ndarray, np.ndarray], ...]:
    reference = np.asarray(seed.pregrasp_joint_positions_rad, dtype=np.float64)
    opened = joint_positions_for_phases(inputs, (0.0, 0.0, 0.0),
                                        reference_joint_positions_rad=reference)
    home = joint_positions_for_phases(inputs, (0.0, 0.0, 0.0),
                                      reference_joint_positions_rad=np.zeros_like(reference))
    target = joint_positions_for_phases(inputs, phases,
                                        reference_joint_positions_rad=reference)
    approach = _approach_states(inputs, seed, target)
    maximum_increment = _control_increment(inputs)
    if not np.isfinite(maximum_increment) or maximum_increment <= 0.0:
        raise ValueError("pregrasp control-step bound must be positive")
    palm_steps = max(1, int(math.ceil(float(np.max(
        np.abs(opened - home))) / maximum_increment)))
    step_count = max(1, int(math.ceil(float(np.max(
        np.abs(target - opened))) / maximum_increment)))
    states = [(f"PALM_FAR_{index:04d}", approach[0][1],
              home + (index / palm_steps) * (opened - home))
              for index in range(palm_steps + 1)]
    for index in range(1, step_count + 1):
        fraction = index / step_count
        sample = tuple(fraction * float(value) for value in phases)
        joints = joint_positions_for_phases(
            inputs, sample, reference_joint_positions_rad=reference)
        states.append((f"PRESHAPE_FAR_{index:04d}", approach[0][1], joints))
    states.extend(approach)
    return tuple(states)
def _precontact_mesh_report(inputs: V2Inputs, state, scene, pad_objects, pad_clearance):
    collision_reason = _first_state_collision(inputs, (state,), scene)
    if scene is None or collision_reason:
        return collision_reason, None
    _self_objects, _nonpad_objects, object_collision, _pairs, _surfaces = scene
    stage, base, joints = state
    transforms = inputs.hand_model.forward_kinematics(joints, base_transform=base)
    collision_request = fcl.CollisionRequest(num_max_contacts=1, enable_contact=False)
    distances = []
    too_close = None
    for pad_name, (link_name, collision_object) in pad_objects.items():
        transform = transforms[link_name]
        collision_object.setTransform(fcl.Transform(transform[:3, :3], transform[:3, 3]))
        intersects = bool(fcl.collide(collision_object, object_collision,
            collision_request, fcl.CollisionResult()))
        if intersects:
            return f"PAD_OBJECT_PRECONTACT_INTERSECTION:{stage}:{pad_name}", None
        if stage == "PREGRASP":
            distance = float(fcl.distance(collision_object, object_collision,
                fcl.DistanceRequest(enable_nearest_points=False), fcl.DistanceResult()))
            distances.append(distance)
            if too_close is None and distance <= pad_clearance:
                too_close = pad_name
    reason = "" if too_close is None else f"PAD_OBJECT_PRECONTACT_DISTANCE:{stage}:{too_close}"
    return reason, None if not distances else tuple(distances)
def fast_filter_pregrasp_paths(
    inputs: V2Inputs,
    variants: tuple[tuple[CandidateSeed, tuple[float, float, float]], ...],
    *, budget_probe_only: bool = False,
) -> tuple[dict[str, object], ...]:
    settings = inputs.config.section("fast_filter")
    if settings["nonpad_collision_policy"] != _NONPAD_POLICY:
        raise ValueError("fast-filter non-PAD collision policy changed")
    scene = _prepare_fcl_scene(inputs)
    pad_objects = {} if scene is None else scene[4]
    configured_maximum = float(inputs.config.section(
        "candidate_generation")["maximum_closure_phase"])
    control_increment = _control_increment(inputs)
    tolerance = float(settings["table_penetration_tolerance_m"])
    pad_clearance = float(inputs.config.section(
        "closure_prediction")["initial_clearance_m"])
    results = []
    shared_state_reports = {}
    for seed, raw_phases in variants:
        phases = tuple(float(value) for value in raw_phases)
        maximum = (configured_maximum if seed.maximum_closure_phase is None
                   else float(seed.maximum_closure_phase))
        remaining = tuple(maximum - value for value in phases)
        reasons = []
        if (maximum <= 0.0 or maximum > configured_maximum
                or any(value <= 0.0 for value in remaining)):
            reasons.append("NO_POSITIVE_THREE_FINGER_REMAINING_CLOSURE")
        if seed.palm_configuration_rad is not None:
            palm_index = inputs.hand_model.independent_joint_names.index("f1j1")
            if not np.isclose(seed.pregrasp_joint_positions_rad[palm_index],
                              seed.palm_configuration_rad, atol=1.0e-12):
                reasons.append("PALM_CONFIGURATION_LOST_IN_PIPELINE")
        if budget_probe_only and phases != (0.0, 0.0, 0.0):
            reasons.append("BUDGET_PROBE_REQUIRES_PHASE_ZERO")
        if reasons:
            states = ()
        elif budget_probe_only:
            reference = np.asarray(seed.pregrasp_joint_positions_rad)
            target = joint_positions_for_phases(
                inputs, phases, reference_joint_positions_rad=reference)
            states = _approach_states(inputs, seed, target)[-1:]
        else:
            states = sampled_pregrasp_path_states(inputs, seed, phases)
        maximum_increment = _maximum_joint_increment(states)
        if maximum_increment > control_increment + 1.0e-12:
            reasons.append("PREGRASP_CONTROL_STEP_INCREMENT_EXCEEDED")
        minimum_table = None
        minimum_link = minimum_stage = ""
        pad_clearances = None
        checked = computed = reused = 0
        for state in states:
            checked += 1
            cache_key = ((seed, state[0])
                         if state[0].startswith("PALM_FAR_") else None)
            if cache_key is not None and cache_key in shared_state_reports:
                reused += 1
                gap, link_name, reason, distances = shared_state_reports[cache_key]
            else:
                computed += 1
                gap, link_name = _state_table_clearance(inputs, state[1], state[2])
                reason = (f"HAND_TABLE_PENETRATION:{state[0]}:{link_name}"
                          if gap is not None and gap < -tolerance else "")
                distances = None
                if not reason:
                    reason, distances = _precontact_mesh_report(
                        inputs, state, scene, pad_objects, pad_clearance)
                if cache_key is not None:
                    shared_state_reports[cache_key] = (
                        gap, link_name, reason, distances)
            if gap is not None and (minimum_table is None or gap < minimum_table):
                minimum_table, minimum_link, minimum_stage = gap, link_name, state[0]
            if distances is not None:
                pad_clearances = distances
            if reason:
                reasons.append(reason)
                break
        results.append({
            "candidate_id": seed.candidate_id,
            "pregrasp_closure_phases": phases,
            "accepted": not reasons,
            "reasons": tuple(reasons),
            "unresolved_checks": (("BUDGET_PROBE_PREGRASP_SNAPSHOT_ONLY",
                "PALM_AND_APPROACH_PATH_NOT_CHECKED_BY_BUDGET_PROBE")
                if budget_probe_only else
                ("FCL_SURFACE_QUERY_CANNOT_PROVE_CONTAINMENT",
                 "HAND_APPROACH_SAMPLED_NOT_CONTINUOUS")),
            "minimum_table_clearance_m": minimum_table,
            "minimum_clearance_link": minimum_link,
            "minimum_clearance_stage": minimum_stage,
            "pregrasp_pad_clearance_by_name_m": None if pad_clearances is None else dict(
                zip(pad_objects, pad_clearances)),
            "checked_state_count": checked,
            "physical_query_state_count": computed,
            "reused_identical_state_count": reused,
            "maximum_joint_increment_rad": maximum_increment,
            "remaining_closure_phase": remaining,
        })
    return tuple(results)
def fast_filter_predictions(
    inputs: V2Inputs, predictions: tuple[ClosurePrediction, ...]
) -> tuple[FastFilterResult, ...]:
    settings = inputs.config.section("fast_filter")
    if settings["nonpad_collision_policy"] != _NONPAD_POLICY:
        raise ValueError("fast-filter non-PAD collision policy changed")
    unresolved = (
        str(settings["arm_ik_policy"]),
        "HAND_SELF_AND_NONPAD_OBJECT_CONTROL_STATES_SAMPLED_NOT_CONTINUOUS",
        "HAND_TABLE_SAMPLED_NOT_CONTINUOUS",
        "ARM_LINK_AND_JOINT_INTERPOLATED_PATH_NOT_FAST_CHECKED",
    )
    collision_scene = _prepare_fcl_scene(inputs)
    results: list[FastFilterResult] = []
    for prediction in predictions:
        reasons = _hard_reasons(inputs, prediction)
        clearance = endpoint_clearance = None
        clearance_link = clearance_stage = ""
        clearance_joints: tuple[float, ...] = ()
        checked_state_count = 0
        first_violation: tuple[float, str, str] | None = None
        maximum_increment = 0.0
        if not reasons:
            states = sampled_sequential_closure_states(
                inputs, prediction.seed, prediction.final_closure_phases)
            endpoint_states = tuple(
                state
                for state in states
                if "CLOSURE_" not in state[0] or state[0].startswith("CONTACT_STOP_")
            )
            tolerance = float(settings["table_penetration_tolerance_m"])
            endpoint_minimum, _ = _state_clearance_summary(inputs, endpoint_states)
            if endpoint_minimum is not None:
                endpoint_clearance = endpoint_minimum[0]
            minimum, first_violation = _state_clearance_summary(
                inputs, states, violation_below_m=-tolerance
            )
            if minimum is not None:
                clearance, clearance_link, clearance_stage, clearance_joints = minimum
            checked_state_count = len(states)
            maximum_increment = _maximum_joint_increment(states)
            if clearance is not None and clearance < -tolerance:
                reasons.append(
                    "INTERMEDIATE_SEQUENTIAL_CLOSURE_HAND_TABLE_SWEEP"
                    if first_violation is not None
                    and first_violation[2].startswith("FINGER_")
                    else "HAND_TABLE_PENETRATION"
                )
            if not reasons:
                collision_reason = _first_state_collision(inputs, states, collision_scene)
                if collision_reason:
                    reasons.append(collision_reason)
        status = "FAST_REJECT" if reasons else "FAST_SURVIVE"
        sweep_pass = (prediction.status == "CLOSURE_SURVIVE"
                      and checked_state_count > 0
                      and (clearance is None or clearance >= -float(
                          settings["table_penetration_tolerance_m"])))
        violation = (None, "", "") if first_violation is None else first_violation
        results.append(
            FastFilterResult(
                candidate_id=prediction.seed.candidate_id,
                status=status,
                reasons=tuple(reasons),
                unresolved_checks=() if reasons else unresolved,
                sequential_closure_sweep_pass=sweep_pass,
                minimum_table_clearance_m=clearance,
                minimum_clearance_link=clearance_link,
                minimum_clearance_finger_stage=clearance_stage,
                minimum_clearance_joint_position_rad=clearance_joints,
                checked_state_count=checked_state_count,
                maximum_joint_increment_rad=maximum_increment,
                endpoint_only_table_clearance_m=endpoint_clearance,
                first_table_violation_clearance_m=violation[0],
                first_table_violation_link=violation[1],
                first_table_violation_finger_stage=violation[2],
            )
        )
    return tuple(results)

__all__ = ["fast_filter_predictions", "fast_filter_pregrasp_paths",
           "sampled_pregrasp_path_states", "sampled_sequential_closure_states"]
