#!/usr/bin/env python3
"""Build object-independent TASK_GRIP_SURFACE meshes for the nail-free hand."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from scipy.spatial import cKDTree

from kcg_connector.grasp.carts_v2.models import (
    joint_positions_for_phases,
    load_v2_inputs,
)
from kcg_connector.grasp.robust.object_model import file_sha256, load_stl_mesh
from kcg_connector.grasp.robust.terminal_pad_source import manifold_edge_face_components
_CONFIG = Path("src/kcg_connector/config/carts_full_palm_search.yaml")
_OBJECT = "current_d38999_26kj61sn_public_spec"
_AUDIT = Path("artifacts/carts_v2/nailfree_height_projected/hand_model_audit")
_OUTPUT = Path("artifacts/carts_v2/nailfree_height_projected/task_grip_surface_audit")
_ROLE_ROOT = Path(
    "artifacts/agent_control/tasks/D38999-AUTONOMOUS-DYNAMIC-CLOSEOUT-V2/"
    "DYN-B-V5-PAD-FORCE-CLOSURE-MULTI-GRASP/BLUE_PAD_SDF_SOURCE_ASSET_V2"
)
_ANCHOR_INDICES = (0, 30, 45, 60, 90)
_PHASE_SAMPLES = (0.0, 0.25, 0.50, 0.75)
_MINIMUM_NORMAL_MOTION_M_PER_PHASE = 1.0e-5
_TERMINAL_BODY_COMPONENT = 0
_LEGACY_PAD_COMPONENT = 75
_JOINT_HOUSING_COMPONENTS = (152, 153, 154)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def _triangle_area(triangles: np.ndarray) -> np.ndarray:
    return 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )


def _removed_and_retained(mesh, audit: dict) -> tuple[np.ndarray, np.ndarray]:
    ranges = [row["source_face_index_range_inclusive"] for row in audit["removed_components"]]
    removed = np.concatenate([
        np.arange(int(low), int(high) + 1, dtype=np.int64) for low, high in ranges
    ])
    expected = int(audit["removed_source_face_count"])
    if len(removed) != expected or len(np.unique(removed)) != expected:
        raise ValueError("nail-free audit has an invalid removed face identity")
    retained = np.ones(len(mesh.faces), dtype=np.bool_)
    retained[removed] = False
    return removed, retained


def _load_link_geometry(root: Path, link: str, audit: dict) -> dict:
    original_path = root / "src/iiwa_description/meshes/hand" / f"{link}.STL"
    nailfree_path = (
        root / "src/iiwa_description/meshes/hand/connector_no_nail"
        / f"{link}_nailfree.stl"
    )
    original, original_provenance = load_stl_mesh(
        original_path, unit="m", orient_outward=False
    )
    if audit["source_sha256"] != original_provenance.source_sha256:
        raise ValueError(f"{link}: nail-free audit source changed")
    removed_indices, retained = _removed_and_retained(original, audit)
    nailfree, nailfree_provenance = load_stl_mesh(
        nailfree_path, unit="m", orient_outward=False
    )
    retained_triangles = original.face_vertices_m[retained]
    if (
        len(nailfree.faces) != len(retained_triangles)
        or not np.array_equal(nailfree.face_vertices_m, retained_triangles)
    ):
        raise ValueError(f"{link}: nail-free source is not the exact retained face stream")
    source_to_nailfree = np.full(len(original.faces), -1, dtype=np.int64)
    source_to_nailfree[np.flatnonzero(retained)] = np.arange(len(nailfree.faces))
    source_components = np.full(len(original.faces), -1, dtype=np.int64)
    for component_id, face_ids in enumerate(manifold_edge_face_components(original.faces)):
        source_components[face_ids] = component_id
    if audit["visual_output_sha256"] != nailfree_provenance.source_sha256:
        raise ValueError(f"{link}: nail-free audit visual changed")
    removed = original.face_vertices_m[removed_indices]
    tip_low = removed.reshape(-1, 3).min(axis=0)
    tip_high = removed.reshape(-1, 3).max(axis=0)
    triangles = nailfree.face_vertices_m
    interface = np.all(
        (triangles.min(axis=1) <= tip_high) & (triangles.max(axis=1) >= tip_low),
        axis=1,
    )
    return {
        "original": original,
        "nailfree": nailfree,
        "original_path": original_path,
        "nailfree_path": nailfree_path,
        "original_sha256": original_provenance.source_sha256,
        "nailfree_sha256": nailfree_provenance.source_sha256,
        "source_to_nailfree": source_to_nailfree,
        "nailfree_source_components": source_components[retained],
        "interface_mask": interface,
        "removed_triangles": removed,
    }


def _role_face_indices(root: Path, geometry: dict, link: str, role: str) -> np.ndarray:
    path = root / _ROLE_ROOT / f"{link}_contact_role_map.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("source_mesh_sha256") != geometry["original_sha256"]:
        raise ValueError(f"{link}: historical semantic source changed")
    rows = [row for row in value.get("faces", ()) if row.get("contact_role") == role]
    centers = np.asarray([row["center_local_m"] for row in rows], dtype=np.float64)
    distance, indices = cKDTree(geometry["original"].face_centroids_m).query(centers, k=1)
    if len(rows) == 0 or np.any(distance > 1.0e-12) or len(np.unique(indices)) != len(rows):
        raise ValueError(f"{link}: historical {role} faces do not map uniquely")
    areas = np.asarray([row["area_m2"] for row in rows])
    normals = np.asarray([row["normal_local"] for row in rows])
    if (
        not np.allclose(geometry["original"].face_areas_m2[indices], areas, atol=1e-15)
        or not np.allclose(geometry["original"].face_normals[indices], normals, atol=1e-12)
    ):
        raise ValueError(f"{link}: historical {role} geometry changed")
    return np.asarray(indices, dtype=np.int64)


def _pad_body_indices(inputs, geometry: dict, pad_name: str) -> np.ndarray:
    pad = next(row for row in inputs.hand_contract.pads if row.name == pad_name)
    with np.load(pad.mesh.absolute_path, allow_pickle=False) as arrays:
        source = np.asarray(arrays["source_face_indices"], dtype=np.int64)
    if not np.array_equal(
        geometry["original"].face_vertices_m[source], pad.points_local_m[pad.faces]
    ):
        raise ValueError(f"{pad_name}: legacy blue PAD lineage changed")
    return source


def _edge_components(mesh, face_ids: np.ndarray, maximum_angle: float) -> list[np.ndarray]:
    selected = {int(value) for value in face_ids}
    edges: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_id in selected:
        face = mesh.faces[face_id]
        for first, second in ((0, 1), (1, 2), (2, 0)):
            edges[tuple(sorted((int(face[first]), int(face[second]))))].append(face_id)
    adjacency: dict[int, list[int]] = defaultdict(list)
    for rows in edges.values():
        if len(rows) != 2:
            continue
        first, second = rows
        angle = np.arccos(np.clip(
            float(mesh.face_normals[first] @ mesh.face_normals[second]), -1.0, 1.0
        ))
        if angle <= maximum_angle:
            adjacency[first].append(second)
            adjacency[second].append(first)
    components = []
    unseen = set(selected)
    while unseen:
        stack = [unseen.pop()]
        component = []
        while stack:
            face = stack.pop()
            component.append(face)
            for neighbor in adjacency[face]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(np.asarray(sorted(component), dtype=np.int64))
    return components


def _derived_patch_rules(geometries: dict, contact_by_link: dict) -> tuple[float, float]:
    angles = []
    largest_areas = []
    for link, geometry in geometries.items():
        original = geometry["original"]
        contact = contact_by_link[link]
        unrestricted = _edge_components(original, contact, np.pi)
        largest_areas.append(max(float(original.face_areas_m2[row].sum()) for row in unrestricted))
        contact_set = set(int(value) for value in contact)
        edges: dict[tuple[int, int], list[int]] = defaultdict(list)
        for face_id in contact_set:
            face = original.faces[face_id]
            for first, second in ((0, 1), (1, 2), (2, 0)):
                edges[tuple(sorted((int(face[first]), int(face[second]))))].append(face_id)
        for rows in edges.values():
            if len(rows) == 2:
                first, second = rows
                angles.append(np.arccos(np.clip(
                    float(original.face_normals[first] @ original.face_normals[second]),
                    -1.0, 1.0,
                )))
    if not angles or not largest_areas:
        raise ValueError("legacy PAD contact patches cannot derive shared rules")
    return float(max(angles) + 1.0e-10), float(min(largest_areas))


def _workspace_centers(inputs, contact_centers: dict, joints: np.ndarray) -> np.ndarray:
    transforms = inputs.hand_model.forward_kinematics(joints)
    points = [
        transforms[link][:3, :3] @ center + transforms[link][:3, 3]
        for link, center in contact_centers.items()
    ]
    return np.mean(points, axis=0)


def _motion_eligible_masks(inputs, geometries: dict, contact_centers: dict) -> dict:
    masks = {
        link: np.zeros(len(row["nailfree"].faces), dtype=np.bool_)
        for link, row in geometries.items()
    }
    pad_names = tuple(pad.name for pad in inputs.hand_contract.pads)
    pad_by_link = {pad.link_name: pad.name for pad in inputs.hand_contract.pads}
    phase_by_pad = {name: index for index, name in enumerate(pad_names)}
    lower = np.asarray([
        inputs.hand_model.joints[name].limit.lower
        for name in inputs.hand_model.independent_joint_names
    ])
    palm_grid = np.linspace(0.0, 1.57, 91)
    for anchor in _ANCHOR_INDICES:
        for phase in _PHASE_SAMPLES:
            reference = lower.copy()
            reference[inputs.hand_model.independent_joint_names.index("f1j1")] = palm_grid[anchor]
            joints = joint_positions_for_phases(
                inputs, (phase, phase, phase), reference_joint_positions_rad=reference
            )
            transforms = inputs.hand_model.forward_kinematics(joints)
            center = _workspace_centers(inputs, contact_centers, joints)
            for link, geometry in geometries.items():
                mesh = geometry["nailfree"]
                transform = transforms[link]
                rotated = mesh.face_centroids_m @ transform[:3, :3].T
                points = rotated + transform[:3, 3]
                normals = mesh.face_normals @ transform[:3, :3].T
                pad_index = phase_by_pad[pad_by_link[link]]
                direction = inputs.closing_directions[pad_index]
                jacobian = inputs.hand_model.geometric_jacobian(link, joints)
                origin_velocity = jacobian[:3] @ direction
                angular_velocity = jacobian[3:] @ direction
                velocity = origin_velocity + np.cross(
                    np.broadcast_to(angular_velocity, rotated.shape), rotated
                )
                inward = center - points
                scale = np.maximum(
                    1.0, np.linalg.norm(inward, axis=1) * np.linalg.norm(velocity, axis=1)
                )
                tolerance = 64.0 * np.finfo(np.float64).eps * scale
                masks[link] |= (
                    (np.einsum("ij,ij->i", normals, inward) > tolerance)
                    & (np.einsum("ij,ij->i", velocity, inward) > tolerance)
                    & (np.einsum("ij,ij->i", normals, velocity)
                       >= _MINIMUM_NORMAL_MOTION_M_PER_PHASE)
                )
    return masks


def _terminal_semantic_whitelist(geometry: dict, blue_faces: np.ndarray) -> tuple[np.ndarray, dict]:
    mesh = geometry["nailfree"]
    component = geometry["nailfree_source_components"]
    if np.any(component[blue_faces] != _LEGACY_PAD_COMPONENT):
        raise ValueError("legacy blue PAD is not confined to its registered component")
    joint_faces = np.flatnonzero(np.isin(component, _JOINT_HOUSING_COMPONENTS))
    if not len(joint_faces):
        raise ValueError("joint housing semantic components are unavailable")
    joint_center = np.average(
        mesh.face_centroids_m[joint_faces], axis=0,
        weights=mesh.face_areas_m2[joint_faces],
    )
    pad_center = np.average(
        mesh.face_centroids_m[blue_faces], axis=0,
        weights=mesh.face_areas_m2[blue_faces],
    )
    distal_axis = pad_center - joint_center
    distal_axis /= np.linalg.norm(distal_axis)
    projection = (mesh.face_centroids_m - joint_center) @ distal_axis
    pad_proximal_projection = float(np.min(projection[blue_faces]))
    body_distal = ((component == _TERMINAL_BODY_COMPONENT)
                   & (projection >= pad_proximal_projection))
    whitelist = (component == _LEGACY_PAD_COMPONENT) | body_distal
    return whitelist, {
        "terminal_body_component": _TERMINAL_BODY_COMPONENT,
        "legacy_pad_component": _LEGACY_PAD_COMPONENT,
        "joint_housing_components_rejected": list(_JOINT_HOUSING_COMPONENTS),
        "joint_center_local_m": joint_center.tolist(),
        "distal_axis_local": distal_axis.tolist(),
        "pad_proximal_projection_m": pad_proximal_projection,
        "distal_body_face_count": int(np.count_nonzero(body_distal)),
    }


def _surface_arrays(mesh, face_ids: np.ndarray, patch_ids: np.ndarray, legacy: np.ndarray) -> dict:
    triangles = mesh.face_vertices_m[face_ids]
    points = triangles.reshape(-1, 3)
    faces = np.arange(len(points), dtype=np.int64).reshape(-1, 3)
    return {
        "points_local_m": points,
        "faces": faces,
        "source_face_indices": face_ids,
        "face_normals_local": mesh.face_normals[face_ids],
        "patch_indices": patch_ids,
        "legacy_blue_pad_face_mask": legacy,
    }


def _render(path: Path, inputs, geometries: dict, allowed: dict, legacy: dict) -> None:
    figure, axes = plt.subplots(5, 3, figsize=(14, 20), constrained_layout=True)
    projections = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))
    lower = np.asarray([inputs.hand_model.joints[n].limit.lower for n in inputs.hand_model.independent_joint_names])
    palm_grid = np.linspace(0.0, 1.57, 91)
    for row, anchor in enumerate(_ANCHOR_INDICES):
        reference = lower.copy()
        reference[inputs.hand_model.independent_joint_names.index("f1j1")] = palm_grid[anchor]
        joints = joint_positions_for_phases(inputs, (0.1, 0.1, 0.1), reference_joint_positions_rad=reference)
        transforms = inputs.hand_model.forward_kinematics(joints)
        for column, (first, second, label) in enumerate(projections):
            axis = axes[row, column]
            bounds = []
            for link, geometry in geometries.items():
                transform = transforms[link]
                triangles = geometry["nailfree"].face_vertices_m
                world = triangles @ transform[:3, :3].T + transform[:3, 3]
                face_ids = allowed[link]
                legacy_mask = legacy[link]
                colors = np.full((len(triangles), 4), (0.55, 0.55, 0.55, 0.16))
                colors[face_ids] = (0.10, 0.65, 0.22, 0.78)
                colors[face_ids[legacy_mask]] = (0.05, 0.35, 0.92, 0.92)
                colors[geometry["interface_mask"]] = (0.95, 0.72, 0.10, 0.75)
                axis.add_collection(PolyCollection(
                    world[:, :, (first, second)], facecolors=colors, edgecolors="none"
                ))
                bounds.append(world.reshape(-1, 3)[:, (first, second)])
            points = np.vstack(bounds)
            low, high = points.min(axis=0), points.max(axis=0)
            margin = max(float(np.ptp(points, axis=0).max()) * 0.04, 0.002)
            axis.set_xlim(low[0] - margin, high[0] + margin)
            axis.set_ylim(low[1] - margin, high[1] + margin)
            axis.set_aspect("equal", adjustable="box")
            axis.set_title(f"q_p index {anchor} ({palm_grid[anchor]:.3f} rad) {label}")
            axis.grid(alpha=0.2)
    figure.suptitle("TASK_GRIP_SURFACE: legacy blue, added inner green, interface yellow")
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main() -> int:
    root = _arguments().repository_root.resolve()
    output = root / _OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    inputs = load_v2_inputs(root, config_path=root / _CONFIG, object_id=_OBJECT)
    hand_audit_path = root / _AUDIT / "NAILFREE_HAND_MODEL_AUDIT.json"
    hand_audit = json.loads(hand_audit_path.read_text(encoding="utf-8"))
    if (
        hand_audit.get("hand_variant") != "CONNECTOR_GRASP_NO_NAIL"
        or hand_audit.get("hardware_authorized") is not False
    ):
        raise ValueError("nail-free hand audit identity is not accepted")
    audit_by_link = {
        Path(row["source_path"]).stem: row for row in hand_audit["links"]
    }
    geometries = {
        pad.link_name: _load_link_geometry(
            root, pad.link_name, audit_by_link[pad.link_name]
        )
        for pad in inputs.hand_contract.pads
    }
    contact_by_link = {
        link: _role_face_indices(root, geometry, link, "PAD_CONTACT_SURFACE")
        for link, geometry in geometries.items()
    }
    contact_centers = {
        link: np.average(
            geometry["original"].face_centroids_m[contact_by_link[link]], axis=0,
            weights=geometry["original"].face_areas_m2[contact_by_link[link]],
        )
        for link, geometry in geometries.items()
    }
    maximum_angle, minimum_patch_area = _derived_patch_rules(
        geometries, contact_by_link
    )
    motion_masks = _motion_eligible_masks(inputs, geometries, contact_centers)
    records, allowed_by_link, legacy_by_link = [], {}, {}
    for pad in inputs.hand_contract.pads:
        link = pad.link_name
        geometry = geometries[link]
        blue_source = _pad_body_indices(inputs, geometry, pad.name)
        blue_new = geometry["source_to_nailfree"][blue_source]
        if np.any(blue_new < 0):
            raise ValueError(f"{link}: legacy blue PAD overlaps removed nail assembly")
        whitelist, whitelist_audit = _terminal_semantic_whitelist(
            geometry, blue_new
        )
        candidate = (motion_masks[link] & whitelist
                     & ~geometry["interface_mask"])
        components = _edge_components(
            geometry["nailfree"], np.flatnonzero(candidate), maximum_angle
        )
        kept = [
            row for row in components
            if float(geometry["nailfree"].face_areas_m2[row].sum()) >= minimum_patch_area
        ]
        if not kept:
            raise ValueError(f"{link}: no connected TASK_GRIP_SURFACE patch survived")
        face_ids = np.concatenate(kept)
        order = np.argsort(face_ids, kind="stable")
        face_ids = face_ids[order]
        patch = np.concatenate([
            np.full(len(row), index, dtype=np.int64) for index, row in enumerate(kept)
        ])[order]
        legacy_mask = np.isin(face_ids, blue_new)
        arrays = _surface_arrays(
            geometry["nailfree"], face_ids, patch, legacy_mask
        )
        npz_path = output / f"{link}_TASK_GRIP_SURFACE_local_m.npz"
        np.savez(npz_path, **arrays)
        old_contact_new = geometry["source_to_nailfree"][contact_by_link[link]]
        records.append({
            "surface_name": f"finger_{pad.finger_name[-1]}_task_grip_surface",
            "pad_name": pad.name,
            "finger_name": pad.finger_name,
            "link_name": link,
            "source_mesh": str(geometry["nailfree_path"].relative_to(root)),
            "source_mesh_sha256": geometry["nailfree_sha256"],
            "source_face_count": int(len(geometry["nailfree"].faces)),
            "surface_npz": str(npz_path.relative_to(root)),
            "surface_npz_sha256": file_sha256(npz_path),
            "task_face_count": int(len(face_ids)),
            "task_area_m2": float(geometry["nailfree"].face_areas_m2[face_ids].sum()),
            "patch_count": int(len(kept)),
            "legacy_blue_pad_face_count": int(np.count_nonzero(legacy_mask)),
            "legacy_pad_contact_face_count": int(np.count_nonzero(np.isin(face_ids, old_contact_new))),
            "added_inner_face_count": int(np.count_nonzero(~legacy_mask)),
            "nail_interface_face_count_rejected": int(np.count_nonzero(geometry["interface_mask"])),
            "motion_compatible_face_count_before_patch_gate": int(np.count_nonzero(candidate)),
            "terminal_semantic_whitelist": whitelist_audit,
            "minimum_normal_motion_m_per_phase": _MINIMUM_NORMAL_MOTION_M_PER_PHASE,
        })
        allowed_by_link[link] = face_ids
        legacy_by_link[link] = legacy_mask
    image = output / "task_grip_surfaces_qp_anchors.png"
    _render(image, inputs, geometries, allowed_by_link, legacy_by_link)
    manifest = {
        "schema_version": "carts_task_grip_surface_v2",
        "hand_variant": "CONNECTOR_GRASP_NO_NAIL",
        "semantic": "TASK_GRIP_SURFACE",
        "hardware_authorized": False,
        "online_control_use_allowed": False,
        "object_specific_selection_used": False,
        "registration_method": "OBJECT_INDEPENDENT_DISTAL_TERMINAL_WHITELIST_AND_FK_MOTION_V2",
        "palm_anchor_indices": list(_ANCHOR_INDICES),
        "closure_phase_samples": list(_PHASE_SAMPLES),
        "maximum_patch_dihedral_rad_from_legacy_pad": maximum_angle,
        "minimum_patch_area_m2_from_legacy_pad": minimum_patch_area,
        "nail_interface_rule": (
            "TRIANGLE_AABB_OVERLAPS_REMOVED_NAIL_ASSEMBLY_AABB_FAIL_CLOSED"
        ),
        "hand_model_audit": str(hand_audit_path.relative_to(root)),
        "hand_model_audit_sha256": file_sha256(hand_audit_path),
        "audit_image": str(image.relative_to(root)),
        "audit_image_sha256": file_sha256(image),
        "links": records,
    }
    manifest_path = output / "TASK_GRIP_SURFACE_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "manifest": str(manifest_path), "sha256": file_sha256(manifest_path),
        "links": records,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
