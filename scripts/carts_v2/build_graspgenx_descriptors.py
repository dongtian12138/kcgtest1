#!/usr/bin/env python3
"""Build and render the fixed, object-independent KCG GraspGenX descriptors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh

from kcg_connector.grasp.carts_v2.gripper_descriptor_builder import (
    build_kcg_graspgenx_descriptors,
    shared_preshape_grid,
)
from kcg_connector.grasp.robust.collision_roster import (
    load_authoritative_collision_link_roster,
)
from kcg_connector.grasp.robust.hand_contract import load_carts_hand_contract
from kcg_connector.grasp.robust.object_model import load_stl_mesh


_MAXIMUM_CLOSURE_PHASE = 0.50
_MAXIMUM_JOINT_INCREMENT_RAD = 0.0015


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--hand-contract",
        type=Path,
        default=Path("src/kcg_connector/config/carts_hand_contact_v1.yaml"),
    )
    parser.add_argument(
        "--collision-roster",
        type=Path,
        default=Path("src/kcg_connector/config/carts_collision_roster_v1.yaml"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _registered_hand_meshes(root: Path, roster, hand) -> dict[str, trimesh.Trimesh]:
    reachable = {hand.base_link, *(joint.child_link for joint in hand.joints.values())}
    meshes: dict[str, trimesh.Trimesh] = {}
    for row in roster.links:
        if row.link_name not in reachable:
            continue
        mesh, provenance = load_stl_mesh(
            row.absolute_path, unit=row.unit, orient_outward=False
        )
        if provenance.source_sha256 != row.sha256:
            raise ValueError(f"collision mesh hash changed for {row.link_name}")
        triangles = np.asarray(mesh.face_vertices_m) * np.asarray(row.scale)
        vertices = triangles.reshape(-1, 3)
        faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
        meshes[row.link_name] = trimesh.Trimesh(
            vertices=vertices, faces=faces, process=False
        )
    if not meshes or hand.base_link not in meshes:
        raise ValueError("registered hand collision mesh coverage is incomplete")
    return meshes


def _self_collision_audits(hand, meshes, maximum_closure_phase: float) -> list[dict]:
    structural = {
        tuple(sorted((joint.parent_link, joint.child_link)))
        for joint in hand.joints.values()
        if joint.parent_link in meshes and joint.child_link in meshes
    }
    manager = trimesh.collision.CollisionManager()
    for name, mesh in meshes.items():
        manager.add_object(name, mesh)
    closure_upper = hand.independent_joint_limits["f1j2"].upper
    close_position = float(maximum_closure_phase) * closure_upper
    step_count = max(1, math.ceil(close_position / _MAXIMUM_JOINT_INCREMENT_RAD))
    positions = np.linspace(0.0, close_position, step_count + 1)
    audits = []
    for spread in shared_preshape_grid(hand):
        first_forbidden = None
        checked = 0
        for closure in positions:
            joints = (spread, closure, closure, closure)
            transforms = hand.forward_kinematics(joints)
            for name in meshes:
                manager.set_transform(name, transforms[name])
            _colliding, pairs = manager.in_collision_internal(return_names=True)
            checked += 1
            forbidden = sorted(
                tuple(sorted(pair))
                for pair in pairs
                if tuple(sorted(pair)) not in structural
            )
            if forbidden:
                first_forbidden = {
                    "closure_position_rad": float(closure),
                    "closure_phase": float(closure / closure_upper),
                    "pairs": [list(pair) for pair in forbidden],
                }
                break
        audits.append(
            {
                "preshape_f1j1_rad": float(spread),
                "pass": first_forbidden is None,
                "checked_state_count": checked,
                "maximum_observed_joint_increment_rad": float(
                    close_position / step_count
                ),
                "first_forbidden_collision": first_forbidden,
                "path": "SYNCHRONOUS_OPEN_TO_CLOSE_AND_REVERSE_SAME_STATES",
                "direct_parent_child_pairs_excluded": True,
            }
        )
    return audits


def _box_lines(offset, extents) -> list[tuple[np.ndarray, np.ndarray]]:
    offset, half = np.asarray(offset), 0.5 * np.asarray(extents)
    corners = np.asarray(
        [offset + half * (x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    )
    return [
        (corners[i], corners[j])
        for i in range(8)
        for j in range(i + 1, 8)
        if np.count_nonzero(np.abs(corners[i] - corners[j]) > 1e-12) == 1
    ]


def _render_descriptor(path: Path, descriptor, hand, meshes) -> None:
    states = (
        ("open", descriptor.open_joint_positions_rad),
        ("half", descriptor.half_joint_positions_rad),
        ("close", descriptor.close_joint_positions_rad),
    )
    figure = plt.figure(figsize=(12, 4), constrained_layout=True)
    transform = descriptor.frame.graspgenx_from_handbase
    for plot_index, (label, joints) in enumerate(states, start=1):
        axis = figure.add_subplot(1, 3, plot_index, projection="3d")
        links = hand.forward_kinematics(joints)
        for mesh_index, (name, mesh) in enumerate(meshes.items()):
            vertices = np.asarray(mesh.vertices)
            stride = max(1, len(vertices) // 900)
            points = vertices[::stride] @ links[name][:3, :3].T + links[name][:3, 3]
            points = points @ transform[:3, :3].T + transform[:3, 3]
            axis.scatter(*points.T, s=0.25, alpha=0.35, color=f"C{mesh_index % 10}")
        if label in ("open", "half"):
            extents = (
                descriptor.open_aabb_extents_m
                if label == "open"
                else descriptor.half_aabb_extents_m
            )
            offset = (
                descriptor.open_aabb_offset_m
                if label == "open"
                else descriptor.half_aabb_offset_m
            )
            for start, end in _box_lines(offset, extents):
                axis.plot(*np.vstack((start, end)).T, color="black", linewidth=0.8)
        axis.quiver(0, 0, 0, 0.04, 0, 0, color="red")
        axis.quiver(0, 0, 0, 0, 0.04, 0, color="green")
        axis.quiver(0, 0, 0, 0, 0, 0.04, color="blue")
        axis.set_title(label)
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=24, azim=-58)
    figure.suptitle(f"{descriptor.descriptor_id}: X closing, Z approach")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> int:
    args = _arguments()
    root = args.repository_root.resolve()
    contract_path = (root / args.hand_contract).resolve()
    roster_path = (root / args.collision_roster).resolve()
    contract = load_carts_hand_contract(contract_path, repository_root=root)
    hand = contract.build_hand_model()
    roster = load_authoritative_collision_link_roster(roster_path, repository_root=root)
    meshes = _registered_hand_meshes(root, roster, hand)
    legacy_audits = _self_collision_audits(hand, meshes, 0.75)
    audits = _self_collision_audits(hand, meshes, _MAXIMUM_CLOSURE_PHASE)
    legal = [row["preshape_f1j1_rad"] for row in audits if row["pass"]]
    descriptors = build_kcg_graspgenx_descriptors(
        contract,
        hand,
        maximum_closure_phase=_MAXIMUM_CLOSURE_PHASE,
        legal_samples_rad=legal,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for descriptor in descriptors:
        image_name = f"{descriptor.descriptor_id}_open_half_close.png"
        image_path = args.output_dir / image_name
        _render_descriptor(image_path, descriptor, hand, meshes)
        rows.append(
            {
                "descriptor_id": descriptor.descriptor_id,
                "preshape_f1j1_rad": descriptor.preshape_f1j1_rad,
                "open_joint_positions_rad": dict(descriptor.open_joint_positions_rad),
                "half_joint_positions_rad": dict(descriptor.half_joint_positions_rad),
                "close_joint_positions_rad": dict(descriptor.close_joint_positions_rad),
                "handbase_from_graspgenx_row_major": (
                    descriptor.frame.handbase_from_graspgenx.ravel().tolist()
                ),
                "graspgenx_from_handbase_row_major": (
                    descriptor.frame.graspgenx_from_handbase.ravel().tolist()
                ),
                "graspgenx_config": descriptor.to_official_config(),
                "render": image_name,
                "render_sha256": _sha256(image_path),
                "aabb_claim": "MODEL_CONDITIONING_ONLY_NOT_COLLISION_PROOF",
            }
        )
    payload = {
        "schema_version": "kcg_graspgenx_descriptors_v1",
        "object_independent": True,
        "hand_contract": str(contract_path.relative_to(root)),
        "hand_contract_sha256": _sha256(contract_path),
        "collision_roster": str(roster_path.relative_to(root)),
        "collision_roster_sha256": _sha256(roster_path),
        "f1j1_grid_rule": "URDF_LIMIT_10_TO_90_PERCENT_9_UNIFORM",
        "maximum_closure_phase": _MAXIMUM_CLOSURE_PHASE,
        "maximum_joint_increment_rad": _MAXIMUM_JOINT_INCREMENT_RAD,
        "self_collision_backend": "python-fcl-0.7.0.8_via_trimesh",
        "legacy_closure_phase_audit": {
            "maximum_closure_phase": 0.75,
            "all_preshapes_pass": all(row["pass"] for row in legacy_audits),
            "audits": legacy_audits,
        },
        "sweep_volume_method": "OFFICIAL_WIZARD_INNER_FINGERTIP_SPACE_AABB",
        "canonical_closing_axis": "+X",
        "canonical_origin_rule": (
            "APPROACH_AXIS_AT_MEAN_PROXIMAL_FINGER_JOINT_PLANE"
        ),
        "self_collision_audits": audits,
        "descriptors": rows,
    }
    destination = args.output_dir / "descriptor_manifest.json"
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"{len(descriptors)} descriptors -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
