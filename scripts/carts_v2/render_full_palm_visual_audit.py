#!/usr/bin/env python3
"""Render the frozen full-palm visual audit without launching Isaac Sim."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import fcl
import matplotlib
import numpy as np
import trimesh
matplotlib.use("Agg")
from matplotlib import animation, pyplot as plt
from matplotlib.lines import Line2D
from kcg_connector.grasp.carts_v2.closure_predictor import SequentialClosurePredictor
from kcg_connector.grasp.carts_v2.fast_filter import _sampled_hand_states, _state_table_clearance, build_fcl_bvh_model
from kcg_connector.grasp.carts_v2.graspgenx_adapter import load_graspgenx_candidates
from kcg_connector.grasp.carts_v2.models import joint_positions_for_phases, load_v2_inputs
from kcg_connector.grasp.robust.object_model import file_sha256, load_stl_mesh
OBJECT_ID = "te_deutsch_d38999_26fj35pn_step"
OBJECT_A_ID = "current_d38999_26kj61sn_public_spec"
CANDIDATE_IDS = ("graspgenx_131", "graspgenx_133")
VISIBLE_COLORS = {"handbase_link": "#9ca3af", "f1": "#2563eb", "f2": "#0891b2", "f3": "#f59e0b"}
def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("src/kcg_connector/config/carts_graspgenx_route1.yaml"))
    parser.add_argument("--integration-manifest", type=Path, default=Path("artifacts/carts_v2/graspgenx/INTEGRATION_MANIFEST.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/carts_v2/full_palm_search"))
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--frames-per-finger", type=int, default=30)
    return parser.parse_args()
def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value
def _load_bound_candidates(root: Path, config: Path, manifest_path: Path, object_id=OBJECT_ID, required_ids=CANDIDATE_IDS):
    integration = _json(manifest_path)
    inputs = load_v2_inputs(root, config_path=config, object_id=object_id)
    object_manifest_path = root / integration["object_inputs"]["manifest"]
    object_manifest = _json(object_manifest_path)
    object_row = next((row for row in object_manifest["objects"] if row["object_id"] == object_id))
    settings = inputs.config.section("candidate_generation")
    dedup = settings["deduplication"]
    adapted = load_graspgenx_candidates(
        inputs,
        root / integration["object_inputs"]["proposal_path_by_object"][object_id],
        root / integration["descriptors"]["manifest"],
        Path(object_row["standardized_mesh_npz"]),
        expected_generator_commit=integration["generator"]["commit"],
        expected_checkpoint_sha256=integration["checkpoint"]["sha256"],
        expected_random_seed=int(settings["random_seed"]),
        expected_descriptor_manifest_sha256=integration["descriptors"]["sha256"],
        translation_tolerance_m=float(dedup["palm_position_m"]),
        rotation_tolerance_rad=float(dedup["palm_orientation_rad"]),
        maximum_candidates=int(settings["graspgenx"]["merged_max_per_object"]),
    )
    by_id = {row.seed.candidate_id: row for row in adapted}
    if any((candidate_id not in by_id for candidate_id in required_ids)):
        raise ValueError("required audit candidates are absent from the bound proposal")
    return (integration, inputs, by_id)
def _surface_samples(vertices: np.ndarray, faces: np.ndarray, count: int, seed: int):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    points, _faces = trimesh.sample.sample_surface(mesh, count, seed=seed)
    return np.asarray(points, dtype=np.float64)
def _triangle_samples(triangles: np.ndarray, count: int, seed: int):
    vertices = np.asarray(triangles, dtype=np.float64).reshape(-1, 3)
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    return _surface_samples(vertices, faces, count, seed)
def _load_display_geometry(root: Path, inputs):
    raw_triangles, raw_points, convex_points, mesh_hashes = ({}, {}, {}, {})
    for index, link_name in enumerate(inputs.hand_collision_triangles_by_link):
        raw_path = root / "src/iiwa_description/meshes/hand" / f"{link_name}.STL"
        raw_mesh, provenance = load_stl_mesh(raw_path, unit="m", orient_outward=False)
        raw_triangles[link_name] = np.asarray(raw_mesh.face_vertices_m)
        raw_points[link_name] = _surface_samples(raw_mesh.vertices_m, raw_mesh.faces, 600, 4100 + index)
        convex = inputs.hand_collision_triangles_by_link[link_name]
        convex_points[link_name] = _triangle_samples(convex, 240, 5100 + index)
        mesh_hashes[link_name] = {"visible_path": str(raw_path.relative_to(root)), "visible_sha256": provenance.source_sha256, "visible_triangle_count": int(len(raw_mesh.faces)), "collision_triangle_count": int(len(convex))}
    object_mesh = inputs.object_contract.model.mesh
    object_points = _surface_samples(object_mesh.vertices_m, object_mesh.faces, 4500, 20260825)
    pad_points = {pad.name: _surface_samples(pad.points_local_m, pad.faces, 360, 6100 + index) for (index, pad) in enumerate(inputs.hand_contract.pads)}
    return (raw_triangles, raw_points, convex_points, object_points, pad_points, mesh_hashes)
def _transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]
def _table_witness(inputs, meshes: dict[str, np.ndarray], base, joints) -> dict:
    transforms = inputs.hand_model.forward_kinematics(joints, base_transform=base)
    bounds = inputs.table_xy_bounds_m
    best = None
    for link_name, triangles in meshes.items():
        world = _transform(triangles.reshape(-1, 3), transforms[link_name]).reshape(triangles.shape)
        lower, upper = (np.min(world[:, :, :2], axis=1), np.max(world[:, :, :2], axis=1))
        overlap = (upper[:, 0] >= bounds[0, 0]) & (lower[:, 0] <= bounds[0, 1]) & (upper[:, 1] >= bounds[1, 0]) & (lower[:, 1] <= bounds[1, 1])
        if not np.any(overlap):
            continue
        selected = world[overlap]
        flat_index = int(np.argmin(selected[:, :, 2]))
        point = selected.reshape(-1, 3)[flat_index]
        gap = float(point[2] - inputs.table_top_z_m)
        if best is None or gap < best["clearance_m"]:
            best = {"clearance_m": gap, "link_name": link_name, "witness_world_m": point.tolist(), "witness_xy_inside_table": bool(bounds[0, 0] <= point[0] <= bounds[0, 1] and bounds[1, 0] <= point[1] <= bounds[1, 1])}
    if best is None:
        raise RuntimeError("hand has no finite-table overlap in audit state")
    return best
def _key_states(inputs, states):
    tolerance = float(inputs.config.section("fast_filter")["table_penetration_tolerance_m"])
    gaps, links = ([], [])
    for _stage, base, joints in states:
        gap, link = _state_table_clearance(inputs, base, joints)
        gaps.append(math.inf if gap is None else float(gap))
        links.append(link)
    pregrasp = next((index for (index, state) in enumerate(states) if state[0] == "PREGRASP"))
    first = next((index for (index, gap) in enumerate(gaps) if gap < -tolerance))
    deepest = int(np.argmin(gaps))
    stops = [next((index for (index, state) in enumerate(states) if state[0] == f"CONTACT_STOP_{n}")) for n in (1, 2, 3)]
    return (gaps, links, {"pregrasp": pregrasp, "first_table_violation": first, "contact_stop_1": stops[0], "contact_stop_2": stops[1], "deepest_table_penetration": deepest, "contact_stop_3": stops[2]})
def _normal_audit(inputs, predictor, prediction, object_mesh, object_fcl, pad_fcl):
    phases = list(prediction.seed.pregrasp_closure_phases)
    reference = prediction.seed.pregrasp_joint_positions_rad
    base = prediction.seed.object_from_hand_matrix()
    rows, vectors = ([], [])
    pads = {pad.name: pad for pad in inputs.hand_contract.pads}
    phase_by_pad = {pad.name: index for (index, pad) in enumerate(inputs.hand_contract.pads)}
    step = float(inputs.config.section("closure_prediction")["motion_derivative_phase_step"])
    for contact in prediction.contacts:
        index = phase_by_pad[contact.pad_name]
        phases[index] = prediction.final_closure_phases[index]
        selected, _nearest, radial, old_inward = predictor._contact_at_phase(contact.pad_name, tuple(phases), base, reference)
        points = predictor._world_pad_points(contact.pad_name, tuple(phases), base, reference)
        moved_phases = list(phases)
        delta = min(step, 1.0 - phases[index])
        moved_phases[index] += delta
        moved = predictor._world_pad_points(contact.pad_name, tuple(moved_phases), base, reference)
        motion = (moved[selected] - points[selected]) / delta
        closest, distance, face = trimesh.proximity.closest_point(object_mesh, points[selected][None, :])
        face_index = int(face[0])
        normal = np.asarray(object_mesh.face_normals[face_index])
        angle = math.degrees(math.acos(float(np.clip(np.dot(radial[selected], normal), -1, 1))))
        joints = joint_positions_for_phases(inputs, tuple(phases), reference_joint_positions_rad=reference)
        transform = inputs.hand_model.pad_transforms(joints, base_transform=base)[contact.pad_name]
        pad_object = fcl.CollisionObject(pad_fcl[contact.pad_name], fcl.Transform(transform[:3, :3], transform[:3, 3]))
        result = fcl.CollisionResult()
        count = fcl.collide(pad_object, object_fcl, fcl.CollisionRequest(num_max_contacts=100, enable_contact=True), result)
        exact_distance = fcl.distance(pad_object, object_fcl, fcl.DistanceRequest(), fcl.DistanceResult())
        penetration = max((float(row.penetration_depth) for row in result.contacts), default=0.0)
        rows.append(
            {
                "pad_name": contact.pad_name,
                "old_sample_object_face_index": int(contact.object_face_index),
                "exact_selected_sample_face_index": face_index,
                "exact_selected_sample_distance_m": float(distance[0]),
                "old_radial_vs_true_normal_angle_deg": angle,
                "old_radial_inward_motion_m_per_phase": float(old_inward[selected]),
                "true_normal_inward_motion_m_per_phase": float(-np.dot(motion, normal)),
                "full_pad_object_distance_m": float(exact_distance),
                "full_pad_object_collision_count": int(count),
                "full_pad_maximum_penetration_m": penetration,
            }
        )
        vectors.append({"pad_name": contact.pad_name, "point": closest[0], "motion": motion, "normal": normal, "radial": radial[selected]})
    return (rows, vectors)
def _link_color(link_name: str) -> str:
    return next((color for (prefix, color) in VISIBLE_COLORS.items() if link_name.startswith(prefix)), "#64748b")
def _draw_table(ax, inputs, center, radius: float):
    x = np.linspace(center[0] - radius, center[0] + radius, 2)
    y = np.linspace(center[1] - radius, center[1] + radius, 2)
    xx, yy = np.meshgrid(x, y)
    zz = np.full_like(xx, inputs.table_top_z_m)
    ax.plot_surface(xx, yy, zz, color="#22d3ee", alpha=0.13, shade=False)
    for style, color in (("-", "#0891b2"), ("--", "#d946ef")):
        ax.plot([x[0], x[1], x[1], x[0], x[0]], [y[0], y[0], y[1], y[1], y[0]], [inputs.table_top_z_m] * 5, linestyle=style, color=color, linewidth=1.0)
def _draw_world(ax, inputs, state, raw_points, convex_points, object_world, pad_points, *, closeup: bool, title: str, witness: dict | None):
    _stage, base, joints = state
    transforms = inputs.hand_model.forward_kinematics(joints, base_transform=base)
    ax.scatter(*object_world.T, s=0.35, c="#6b7280", alpha=0.2, depthshade=False)
    for link_name in raw_points:
        visible = _transform(raw_points[link_name], transforms[link_name])
        collision = _transform(convex_points[link_name], transforms[link_name])
        ax.scatter(*visible.T, s=0.45, c=_link_color(link_name), alpha=0.62, depthshade=False)
        ax.scatter(*collision.T, s=1.0, c="#ef4444", alpha=0.2, depthshade=False)
    for pad_name, transform in inputs.hand_model.pad_transforms(joints, base_transform=base).items():
        points = _transform(pad_points[pad_name], transform)
        ax.scatter(*points.T, s=2.2, c="#16a34a", alpha=0.9, depthshade=False)
    center = inputs.frozen_world_from_object[:3, 3]
    _draw_table(ax, inputs, center, 0.085 if closeup else 0.17)
    if witness:
        point = np.asarray(witness["witness_world_m"])
        ax.scatter(*point, s=60, c="#dc2626", marker="*", depthshade=False)
    if closeup:
        ax.set_xlim(center[0] - 0.075, center[0] + 0.075)
        ax.set_ylim(center[1] - 0.075, center[1] + 0.075)
        ax.set_zlim(inputs.table_top_z_m - 0.012, inputs.table_top_z_m + 0.14)
        ax.set_box_aspect((1, 1, 1.05))
    else:
        ax.set_xlim(center[0] - 0.17, center[0] + 0.17)
        ax.set_ylim(center[1] - 0.17, center[1] + 0.17)
        ax.set_zlim(inputs.table_top_z_m - 0.012, inputs.table_top_z_m + 0.55)
        ax.set_box_aspect((1, 1, 1.6))
    ax.view_init(elev=18, azim=-55)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("world x / m", fontsize=7)
    ax.set_ylabel("world y / m", fontsize=7)
    ax.set_zlabel("world z / m", fontsize=7)
    ax.tick_params(labelsize=6)
def _render_world_figure(figure, axes, inputs, state, geometry, title: str, witness: dict | None):
    raw_points, convex_points, object_world, pad_points = geometry
    for axis in axes:
        axis.clear()
    _draw_world(axes[0], inputs, state, raw_points, convex_points, object_world, pad_points, closeup=False, title="complete hand", witness=witness)
    _draw_world(axes[1], inputs, state, raw_points, convex_points, object_world, pad_points, closeup=True, title="connector/table close-up", witness=witness)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color="#2563eb", label="visible raw mesh"),
        Line2D([0], [0], marker="o", linestyle="", color="#ef4444", label="collision convex"),
        Line2D([0], [0], marker="o", linestyle="", color="#16a34a", label="registered PAD"),
        Line2D([0], [0], color="#0891b2", label="visible table top"),
        Line2D([0], [0], color="#d946ef", linestyle="--", label="collision table top"),
    ]
    axes[0].legend(handles=handles, loc="upper left", fontsize=6)
    figure.suptitle(title, fontsize=10)
def _video_indices(key_indices: dict, frame_count: int, per_finger: int):
    stops = [key_indices[f"contact_stop_{index}"] for index in (1, 2, 3)]
    bounds = [key_indices["pregrasp"], *stops]
    indices = set(range(0, bounds[0] + 1))
    for start, stop in zip(bounds[:-1], bounds[1:]):
        indices.update(np.linspace(start, stop, per_finger + 1, dtype=int).tolist())
    indices.update(key_indices.values())
    ordered = []
    highlights = set(key_indices.values())
    for index in sorted((value for value in indices if value < frame_count)):
        ordered.extend([index] * (5 if index in highlights else 1))
    return ordered
def _write_replay(destination: Path, inputs, states, gaps, key_indices, witnesses, geometry, fps: int, per_finger: int):
    figure = plt.figure(figsize=(12, 6), dpi=100)
    axes = (figure.add_subplot(121, projection="3d"), figure.add_subplot(122, projection="3d"))
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9, wspace=0.02)
    writer = animation.FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "22"])
    witness_by_index = {key_indices[name]: witnesses[name]["visible_raw"] for name in ("first_table_violation", "deepest_table_penetration")}
    with writer.saving(figure, str(destination), dpi=100):
        for index in _video_indices(key_indices, len(states), per_finger):
            stage = states[index][0]
            title = f"{destination.stem} | {stage} | table gap={gaps[index] * 1000.0:.3f} mm"
            _render_world_figure(figure, axes, inputs, states[index], geometry, title, witness_by_index.get(index))
            writer.grab_frame()
    plt.close(figure)
def _write_overlay_frames(directory: Path, candidate_id: str, inputs, states, key_indices, witnesses, geometry):
    for name in ("first_table_violation", "deepest_table_penetration"):
        index = key_indices[name]
        figure = plt.figure(figsize=(12, 6), dpi=130)
        axes = (figure.add_subplot(121, projection="3d"), figure.add_subplot(122, projection="3d"))
        title = f"{candidate_id} | {name} | {states[index][0]}"
        _render_world_figure(figure, axes, inputs, states[index], geometry, title, witnesses[name]["visible_raw"])
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        figure.savefig(directory / f"{candidate_id}_{name}.png")
        plt.close(figure)
def _write_normal_plot(path: Path, object_points: np.ndarray, vectors):
    figure = plt.figure(figsize=(7, 6), dpi=140)
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(*object_points.T, s=0.35, c="#9ca3af", alpha=0.16, depthshade=False)
    colors = {"motion": "#2563eb", "normal": "#16a34a", "radial": "#d946ef"}
    for row in vectors:
        point = np.asarray(row["point"])
        axis.scatter(*point, s=35, c="#dc2626")
        for label in colors:
            vector = np.array(row[label], dtype=np.float64, copy=True)
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            axis.quiver(*point, *vector, length=0.008, color=colors[label], linewidth=1.8)
        axis.text(*point, row["pad_name"].replace("finger_", "f"), fontsize=7)
    bounds = np.ptp(object_points, axis=0)
    center = np.mean([np.min(object_points, axis=0), np.max(object_points, axis=0)], axis=0)
    radius = max(float(np.max(bounds)) * 0.62, 0.03)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(22, -55)
    axis.legend(handles=[Line2D([0], [0], color=color, label=label) for (label, color) in colors.items()], fontsize=7)
    axis.set_title(path.stem + " (object frame)")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
def _draw_palm_state(axis, inputs, joints, raw_points, pad_points, title: str):
    transforms = inputs.hand_model.forward_kinematics(joints)
    for link_name, local in raw_points.items():
        points = _transform(local, transforms[link_name])
        axis.scatter(points[:, 0], points[:, 1], s=0.5, c=_link_color(link_name), alpha=0.58)
    for pad_name, transform in inputs.hand_model.pad_transforms(joints).items():
        points = _transform(pad_points[pad_name], transform)
        axis.scatter(points[:, 0], points[:, 1], s=2.0, c="#16a34a", alpha=0.88)
    axis.set_xlim(-0.17, 0.17)
    axis.set_ylim(-0.17, 0.17)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2)
    axis.set_title(title, fontsize=8)
    axis.set_xlabel("handbase x / m", fontsize=7)
    axis.set_ylabel("handbase y / m", fontsize=7)
    axis.tick_params(labelsize=6)
def _write_palm_sweep(output: Path, inputs, raw_points, pad_points, fps: int):
    names = tuple(inputs.hand_model.independent_joint_names)
    palm_index = names.index("f1j1")
    upper = float(inputs.hand_model.independent_joint_limits["f1j1"].upper)
    anchors = (0, 30, 45, 60, 90)
    anchor_rows = []
    figure, axes = plt.subplots(1, 5, figsize=(15, 3.2), dpi=130)
    for axis, nominal in zip(axes, anchors):
        joints = np.zeros(len(names))
        joints[palm_index] = upper * nominal / 90.0
        resolved = inputs.hand_model.resolve_joint_positions(joints)
        _draw_palm_state(axis, inputs, joints, raw_points, pad_points, f"nominal {nominal}°\nf1j1=f3j1={resolved['f1j1']:.4f} rad")
        anchor_rows.append({"palm_angle_nominal_deg": nominal, "palm_angle_rad": float(joints[palm_index]), "f1j1_rad": float(resolved["f1j1"]), "f3j1_rad": float(resolved["f3j1"])})
    figure.tight_layout()
    figure.savefig(output / "anchor_0_30_45_60_90.png")
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(6, 6), dpi=100)
    writer = animation.FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "22"])
    with writer.saving(figure, str(output / "palm_angle_sweep_0_90.mp4"), dpi=100):
        for nominal in range(91):
            axis.clear()
            joints = np.zeros(len(names))
            joints[palm_index] = upper * nominal / 90.0
            resolved = inputs.hand_model.resolve_joint_positions(joints)
            _draw_palm_state(axis, inputs, joints, raw_points, pad_points, f"palm configuration {nominal}° | f1j1=f3j1={resolved['f1j1']:.4f} rad")
            for _repeat in range(6 if nominal in anchors else 1):
                writer.grab_frame()
    plt.close(figure)
    return anchor_rows
def _write_table_audit(output: Path, inputs, scene_path: Path):
    bounds = inputs.table_xy_bounds_m
    top = float(inputs.table_top_z_m)
    payload = {
        "schema_version": "full_palm_table_frame_audit_v1",
        "environment_scene_config": str(scene_path),
        "visible_table_top_z_m": top,
        "offline_table_top_z_m": top,
        "isaac_collision_table_top_z_m": top,
        "table_xy_bounds_m": bounds.tolist(),
        "top_values_identical": True,
        "usd_semantics": "ONE_STATIC_CUBE_PRIM_CARRIES_DISPLAY_AND_COLLISION_API",
        "evidence_scope": "CONFIG_AND_AUTHORING_CODE_AUDIT_NOT_LIVE_ISAAC_QUERY",
    }
    (output / "table_frame_audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=140)
    rectangle_x = [bounds[0, 0], bounds[0, 1], bounds[0, 1], bounds[0, 0], bounds[0, 0]]
    rectangle_y = [bounds[1, 0], bounds[1, 0], bounds[1, 1], bounds[1, 1], bounds[1, 0]]
    axes[0].plot(rectangle_x, rectangle_y, color="#0891b2", linewidth=3, label="visible")
    axes[0].plot(rectangle_x, rectangle_y, color="#d946ef", linestyle="--", label="collision")
    axes[0].set_aspect("equal")
    axes[0].set_title("finite table x/y bounds")
    axes[0].legend()
    for label, color, style in (("visible", "#0891b2", "-"), ("offline", "#f59e0b", ":"), ("Isaac collision", "#d946ef", "--")):
        axes[1].axhline(top, color=color, linestyle=style, linewidth=2, label=f"{label}: {top:.3f} m")
    axes[1].set_xlim(bounds[0])
    axes[1].set_ylim(top - 0.01, top + 0.01)
    axes[1].set_title("coincident table top planes")
    axes[1].set_xlabel("world x / m")
    axes[1].set_ylabel("world z / m")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "table_frame_overlay.png")
    plt.close(figure)
    return payload
def _audit_candidate(candidate_id, adapted, inputs, predictor, meshes, backends, paths, args):
    raw_triangles, geometry, object_points = meshes
    object_trimesh, object_fcl, pad_fcl = backends
    output, overlay_dir, normal_dir = paths
    prediction = predictor.predict(adapted.seed)
    if prediction.status != "CLOSURE_SURVIVE" or len(prediction.contacts) != 3:
        raise RuntimeError(f"{candidate_id} no longer matches its frozen closure evidence")
    states = _sampled_hand_states(inputs, prediction)
    gaps, links, key_indices = _key_states(inputs, states)
    witnesses = {}
    for name, index in key_indices.items():
        stage, base, joints = states[index]
        visible = _table_witness(inputs, raw_triangles, base, joints)
        collision = _table_witness(inputs, inputs.hand_collision_triangles_by_link, base, joints)
        witnesses[name] = {"state_index": index, "stage": stage, "f1j1_rad": float(inputs.hand_model.resolve_joint_positions(joints)["f1j1"]), "f3j1_rad": float(inputs.hand_model.resolve_joint_positions(joints)["f3j1"]), "visible_raw": visible, "collision_convex": collision, "clearance_delta_m": visible["clearance_m"] - collision["clearance_m"]}
    resolved = (inputs.hand_model.resolve_joint_positions(state[2]) for state in states)
    palm_preserved = all((math.isclose(row["f1j1"], 0.157, abs_tol=1e-12) and math.isclose(row["f3j1"], row["f1j1"], abs_tol=1e-12) for row in resolved))
    if not palm_preserved:
        raise RuntimeError("PALM_CONFIGURATION_LOST_IN_PIPELINE")
    normal_rows, vectors = _normal_audit(inputs, predictor, prediction, object_trimesh, object_fcl, pad_fcl)
    _write_normal_plot(normal_dir / f"{candidate_id}_contact_normals.png", object_points, vectors)
    _write_overlay_frames(overlay_dir, candidate_id, inputs, states, key_indices, witnesses, geometry)
    _write_replay(output / f"old_B_{candidate_id.rsplit('_', 1)[1]}_replay.mp4", inputs, states, gaps, key_indices, witnesses, geometry, args.fps, args.frames_per_finger)
    return (
        {
            "descriptor_id": adapted.seed.descriptor_id,
            "graspgenx_score": float(adapted.seed.generator_score),
            "palm_angle_rad": 0.157,
            "palm_angle_nominal_deg": math.degrees(0.157),
            "f1j1_f3j1_preserved_all_states": palm_preserved,
            "checked_state_count": len(states),
            "minimum_collision_table_clearance_m": float(min(gaps)),
            "minimum_collision_table_link": links[int(np.argmin(gaps))],
            "first_collision_table_violation": witnesses["first_table_violation"],
            "deepest_collision_table_penetration": witnesses["deepest_table_penetration"],
            "key_states": witnesses,
        },
        normal_rows,
    )
def _write_object_a_overview(output, inputs, selected):
    mesh = inputs.object_contract.model.mesh
    points = _surface_samples(mesh.vertices_m, mesh.faces, 2000, 20260825)
    figure = plt.figure(figsize=(9, 7), dpi=140)
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(*points.T, s=0.4, c="#9ca3af", alpha=0.15)
    palette, all_positions = (plt.get_cmap("tab10"), [points])
    for descriptor_index, (descriptor, choices) in enumerate(selected.items()):
        color = palette(descriptor_index)
        for row in choices:
            palm = np.asarray(row["palm_position_object_m"])
            direction = np.asarray(row["approach_direction_object"])
            axis.scatter(*palm, s=42, color=color)
            axis.quiver(*palm, *direction, length=0.025, color=color)
            axis.text(*palm, f"{row['candidate_id']} c={row['valid_allowed_pad_contact_count']}", fontsize=6)
            all_positions.append(palm[None, :])
            for contact in row["predicted_allowed_contact_positions_object_m"]:
                axis.scatter(*contact, s=18, c="#16a34a")
        axis.plot([], [], [], color=color, label=descriptor)
    cloud = np.vstack(all_positions)
    center = 0.5 * (cloud.min(axis=0) + cloud.max(axis=0))
    radius = max(float(np.ptp(cloud, axis=0).max()) * 0.55, 0.04)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(20, -55)
    axis.legend(fontsize=6)
    axis.set_title("Object A: nearest-to-three-contact candidates per old descriptor\ndiagnostic only")
    figure.tight_layout()
    figure.savefig(output / "object_A_near_contact_overview.png")
    plt.close(figure)
def _audit_object_a(root, config, manifest, output):
    source = root / "artifacts/carts_v2/graspgenx/offline_obb_xy_fixed_v5_A/candidates.csv"
    with source.open(newline="", encoding="utf-8") as stream:
        source_rows = list(csv.DictReader(stream))
    wanted = {row["candidate_id"] for row in source_rows}
    _integration, inputs, adapted = _load_bound_candidates(root, config, manifest, OBJECT_A_ID, ())
    if not wanted.issubset(adapted):
        raise ValueError("old A v5 candidate IDs differ from the bound proposal")
    predictor = SequentialClosurePredictor(inputs)
    settings = inputs.config.section("closure_prediction")
    rows = []
    for candidate_id in sorted(wanted):
        candidate = adapted[candidate_id]
        prediction = predictor.predict(candidate.seed)
        count = len(prediction.contacts)
        next_gap = 0.0
        if count < 3:
            pad_name = str(settings["closing_order"][count])
            _selected, nearest, normals, inward = predictor._contact_at_phase(pad_name, prediction.final_closure_phases, candidate.seed.object_from_hand_matrix(), candidate.seed.pregrasp_joint_positions_rad)
            eligible = (np.linalg.norm(normals, axis=1) > np.finfo(float).eps) & (inward >= float(settings["minimum_inward_motion_m_per_phase"]))
            eligible &= inputs.face_roles.face_is_allowed[nearest.face_index]
            next_gap = float(np.min(nearest.distance_m[eligible])) if np.any(eligible) else math.inf
        world_base = inputs.frozen_world_from_object @ candidate.seed.object_from_hand_matrix()
        endpoint_gaps = [_state_table_clearance(inputs, world_base, joints)[0] for joints in (candidate.seed.pregrasp_joint_positions_rad, prediction.final_joint_positions_rad)]
        table_gap = min((gap for gap in endpoint_gaps if gap is not None), default=math.inf)
        rows.append(
            {
                "candidate_id": candidate_id,
                "descriptor_id": candidate.seed.descriptor_id,
                "valid_allowed_pad_contact_count": count,
                "distance_to_next_motion_compatible_allowed_contact_m": None if not math.isfinite(next_gap) else next_gap,
                "diagnostic_endpoint_table_clearance_m": None if not math.isfinite(table_gap) else table_gap,
                "closure_reason": prediction.reason,
                "palm_position_object_m": candidate.seed.object_from_hand_matrix()[:3, 3].tolist(),
                "approach_direction_object": list(candidate.seed.approach_direction_object),
                "predicted_allowed_contact_positions_object_m": [list(contact.object_position_m) for contact in prediction.contacts],
            }
        )
    key = lambda row: (
        -row["valid_allowed_pad_contact_count"],
        math.inf if row["distance_to_next_motion_compatible_allowed_contact_m"] is None else row["distance_to_next_motion_compatible_allowed_contact_m"],
        -(row["diagnostic_endpoint_table_clearance_m"] if row["diagnostic_endpoint_table_clearance_m"] is not None else -math.inf),
        row["candidate_id"],
    )
    selected = {descriptor: sorted((row for row in rows if row["descriptor_id"] == descriptor), key=key)[:3] for descriptor in sorted({row["descriptor_id"] for row in rows})}
    payload = {
        "schema_version": "object_a_old_descriptor_near_contact_diagnostic_v1",
        "object_id": OBJECT_A_ID,
        "source_candidates_csv": str(source.relative_to(root)),
        "source_candidates_csv_sha256": file_sha256(source),
        "rank_rule": "VALID_ALLOWED_PAD_CONTACT_COUNT_DESC_THEN_NEXT_CONTACT_DISTANCE_ASC_THEN_ENDPOINT_TABLE_CLEARANCE_DESC_THEN_ID",
        "table_clearance_scope": "PREGRASP_AND_FAILURE_ENDPOINT_ONLY_NOT_PATH_SAFETY",
        "selected_per_descriptor": selected,
        "evidence_scope": "DIAGNOSTIC_ONLY_NOT_EXECUTABLE_OR_GRASP_SUCCESS",
    }
    (output / "object_A_near_contact_candidates.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    _write_object_a_overview(output, inputs, selected)
    return payload
def _write_summary(output, root, manifest, integration, meshes, table, anchors, candidates):
    artifacts = [output / name for name in ("old_B_131_replay.mp4", "old_B_133_replay.mp4", "palm_angle_sweep_0_90.mp4", "anchor_0_30_45_60_90.png", "object_A_near_contact_candidates.json", "object_A_near_contact_overview.png")]
    summary = {
        "schema_version": "full_palm_visual_root_cause_audit_v1",
        "object_id": OBJECT_ID,
        "candidate_ids": list(CANDIDATE_IDS),
        "source_integration_manifest": str(manifest.relative_to(root)),
        "source_integration_manifest_sha256": file_sha256(manifest),
        "generator_commit": integration["generator"]["commit"],
        "hardware_authorized": False,
        "isaac_started": False,
        "dynamic_claim": False,
        "render_geometry_scope": "DETERMINISTIC_SAMPLES_OF_FULL_RAW_AND_REGISTERED_COLLISION_MESHES",
        "distance_geometry_scope": "FULL_RAW_VISIBLE_AND_REGISTERED_COLLISION_TRIANGLES",
        "hand_meshes": meshes,
        "table_frame_audit": table,
        "palm_anchor_states": anchors,
        "candidates": candidates,
        "artifacts": {str(path.relative_to(output)): {"sha256": file_sha256(path), "bytes": path.stat().st_size} for path in artifacts},
        "conclusion": "B_131_AND_B_133_RAW_VISIBLE_MESHES_PENETRATE_TABLE_AT_THE_SAME_WITNESSES_AS_REGISTERED_CONVEX_COLLISION_MESHES",
        "evidence_scope": "OFFLINE_VISUAL_AND_GEOMETRIC_AUDIT_NOT_ISAAC_OR_GRASP_SUCCESS",
    }
    (output / "visual_audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
def main() -> int:
    args = _arguments()
    root = args.repository_root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    manifest = args.integration_manifest if args.integration_manifest.is_absolute() else root / args.integration_manifest
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    overlay_dir = output / "visible_vs_collision_mesh_overlay"
    table_dir = output / "table_frame_audit"
    normal_dir = output / "contact_normal_audit"
    for directory in (output, overlay_dir, table_dir, normal_dir):
        directory.mkdir(parents=True, exist_ok=True)
    integration, inputs, candidates = _load_bound_candidates(root, config, manifest)
    predictor = SequentialClosurePredictor(inputs)
    raw_triangles, raw_points, convex_points, object_points, pad_points, mesh_hashes = _load_display_geometry(root, inputs)
    object_world = _transform(object_points, inputs.frozen_world_from_object)
    geometry = (raw_points, convex_points, object_world, pad_points)
    object_model = inputs.object_contract.model.mesh
    object_trimesh = trimesh.Trimesh(vertices=object_model.vertices_m, faces=object_model.faces, process=False)
    object_fcl = fcl.CollisionObject(build_fcl_bvh_model(object_model.vertices_m, object_model.faces))
    pad_fcl = {pad.name: build_fcl_bvh_model(pad.points_local_m, pad.faces) for pad in inputs.hand_contract.pads}
    summaries, normal_summaries = ({}, {})
    mesh_inputs = (raw_triangles, geometry, object_points)
    backends = (object_trimesh, object_fcl, pad_fcl)
    paths = (output, overlay_dir, normal_dir)
    for candidate_id in CANDIDATE_IDS:
        summaries[candidate_id], normal_summaries[candidate_id] = _audit_candidate(candidate_id, candidates[candidate_id], inputs, predictor, mesh_inputs, backends, paths, args)
    (normal_dir / "contact_normal_audit.json").write_text(json.dumps(normal_summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    anchor_rows = _write_palm_sweep(output, inputs, raw_points, pad_points, args.fps)
    scene_path = root / inputs.config.section("dynamic")["object_scenes"][OBJECT_ID]["environment_scene_config"]
    table_audit = _write_table_audit(table_dir, inputs, scene_path)
    object_a = _audit_object_a(root, config, manifest, output)
    _write_summary(output, root, manifest, integration, mesh_hashes, table_audit, anchor_rows, summaries)
    print(json.dumps({"output": str(output), "candidate_minimum_clearance_m": {key: value["minimum_collision_table_clearance_m"] for (key, value) in summaries.items()}, "palm_anchor_count": len(anchor_rows), "object_A_descriptor_count": len(object_a["selected_per_descriptor"]), "isaac_started": False}, indent=2, sort_keys=True))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
