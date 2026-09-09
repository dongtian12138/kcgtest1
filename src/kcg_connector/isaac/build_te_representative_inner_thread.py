#!/usr/bin/env python3
"""Cut a standard-bounded female thread into the incomplete TE customer CAD.

This is offline geometry authoring. It produces a representative mating region,
not a claim to reproduce the supplier's undisclosed thread approach or detent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh

from build_te_free_split_plug import _close_hidden_planar_boundaries


def build(repository: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    source = repository / "artifacts/carts_grasp/CARTS_GRASP_V1/object_models/te_j35/coupling_nut_visual_mesh.npz"
    data = np.load(source)
    vertices, faces, caps = _close_hidden_planar_boundaries(data["vertices_m"], data["faces"])
    original = trimesh.Trimesh(vertices, faces, process=True)
    if not original.is_volume:
        raise ValueError("the existing closed source nut is not a positive watertight volume")
    pitch, lead = 0.00254, 0.00762
    minor_radius = 0.0199644  # Existing source bore; within the DD minor limits.
    pitch_radius = 1.597 * 0.0254 / 2.0  # Midpoint of 1.591..1.603 inch.
    major_radius = 1.639 * 0.0254 / 2.0  # Midpoint of 1.629..1.649 inch.
    # The groove is a modified 60-degree stub. At the pitch radius its width
    # is half the pitch. Limits of size constrain the three radii independently.
    n_theta = 720
    theta = np.arange(n_theta) * (2.0 * np.pi / n_theta)
    z = np.unique(np.r_[np.linspace(-0.0150, -0.0010, 561), 0.0030])
    phase = (z[:, None] - lead * theta[None, :] / (2.0 * np.pi) + pitch / 2.0) % pitch - pitch / 2.0
    radius = np.clip(pitch_radius + np.sqrt(3.0) * (pitch / 4.0 - np.abs(phase)),
                     minor_radius, major_radius)
    # Representative 0.20 mm rear runout, entirely forward of the original
    # z=-15.0114 mm internal shoulder. The source shoulder cap is retained.
    taper = np.clip((z + 0.0150) / 0.0002, 0.0, 1.0)
    radius = (minor_radius - 0.0002) + taper[:, None] * (radius - (minor_radius - 0.0002))
    cutter_vertices = np.stack((radius * np.cos(theta), radius * np.sin(theta),
                               np.broadcast_to(z[:, None], radius.shape)), axis=-1).reshape(-1, 3)
    lower = (np.arange(len(z) - 1)[:, None] * n_theta + np.arange(n_theta)[None, :]).ravel()
    next_lower = (np.arange(len(z) - 1)[:, None] * n_theta
                  + (np.arange(n_theta)[None, :] + 1) % n_theta).ravel()
    cutter_faces = np.vstack((np.column_stack((lower, next_lower, next_lower + n_theta)),
                              np.column_stack((lower, next_lower + n_theta, lower + n_theta))))
    low_center, high_center = len(cutter_vertices), len(cutter_vertices) + 1
    cutter_vertices = np.vstack((cutter_vertices, [0, 0, z[0]], [0, 0, z[-1]]))
    a = np.arange(n_theta)
    b = (a + 1) % n_theta
    top = (len(z) - 1) * n_theta
    cutter_faces = np.vstack((cutter_faces, np.column_stack((b, a, np.full(n_theta, low_center))),
                              np.column_stack((top + a, top + b, np.full(n_theta, high_center)))))
    cutter = trimesh.Trimesh(cutter_vertices, cutter_faces, process=True)
    if not cutter.is_volume:
        raise ValueError("thread cutting tool is not a positive watertight volume")
    threaded = trimesh.boolean.difference([original, cutter], engine="manifold")
    # The boolean backend can retain coincident vertex indices and zero-area
    # triangles while reporting topological watertightness. PhysX welds equal
    # positions, so canonicalize those exact duplicates before cooking. This
    # removes no surface and moves no vertex.
    unique_vertices, inverse = np.unique(threaded.vertices, axis=0, return_inverse=True)
    remapped_faces = inverse[threaded.faces]
    keep = ((remapped_faces[:, 0] != remapped_faces[:, 1])
            & (remapped_faces[:, 1] != remapped_faces[:, 2])
            & (remapped_faces[:, 2] != remapped_faces[:, 0]))
    cleanup = {"exact_duplicate_vertex_count": len(threaded.vertices) - len(unique_vertices),
               "zero_area_repeated_index_faces_removed": int(np.sum(~keep)),
               "vertex_position_change_m": 0.0, "nonzero_surface_faces_removed": 0}
    threaded = trimesh.Trimesh(unique_vertices, remapped_faces[keep], process=False)
    if np.any(threaded.area_faces <= 0):
        raise ValueError("thread subtraction retained degenerate surface triangles")
    if not threaded.is_volume or len(threaded.split(only_watertight=False)) != 1:
        raise ValueError("thread subtraction did not produce one closed nut")
    if not 0 < threaded.volume < original.volume:
        raise ValueError("thread subtraction has an invalid removed volume")
    if np.max(np.abs(threaded.bounds - original.bounds)) > 1e-8:
        raise ValueError("thread subtraction changed the original exterior bounds")
    out = output / "coupling_nut_standard_inner_thread.npz"
    np.savez_compressed(out, vertices_m=np.asarray(threaded.vertices),
                        faces=np.asarray(threaded.faces, dtype=np.int32))
    threaded.export(output / "coupling_nut_standard_inner_thread.stl")
    result = {
        "schema_version": "te_representative_internal_thread_v1",
        "status": "GEOMETRY_ONLY_NOT_YET_SIMULATED",
        "source_npz": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_modified": False, "output_mesh": str(out),
        "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "standard": "MIL-DTL-38999N w/AMENDMENT 2, 27 July 2026, Figure 3, printed pages 91-93",
        "standard_local_copy": "artifacts/kcg_connector/reference/full_assembly_20260905/MIL-DTL-38999N_Amendment2_20260727.pdf",
        "shell_size": 25, "thread_starts": 3, "pitch_m": pitch, "lead_m": lead,
        "minor_diameter_m": 2 * minor_radius, "pitch_diameter_m": 2 * pitch_radius,
        "major_diameter_m": 2 * major_radius, "included_flank_angle_deg": 60,
        "source_frame_helix_relation": "z - lead*theta/(2*pi) = constant",
        "phase_zero_convention": "arbitrary representative thread datum; not tuned to online start pose",
        "supplier_unverified_details": ["exact female thread phase", "approach relief", "rear runout", "self-locking detent", "seal constitutive behavior"],
        "representative_rear_runout_m": 0.0002,
        "original_hidden_interface_caps_retained": caps,
        "watertight": bool(threaded.is_watertight), "positive_volume": bool(threaded.is_volume),
        "original_bounds_m": original.bounds.tolist(), "threaded_bounds_m": threaded.bounds.tolist(),
        "original_volume_m3": float(original.volume), "threaded_volume_m3": float(threaded.volume),
        "removed_volume_m3": float(original.volume - threaded.volume),
        "vertex_count": len(threaded.vertices), "triangle_count": len(threaded.faces),
        "exact_degenerate_cleanup": cleanup,
        "angular_sampling_deg": 360.0 / n_theta, "main_axial_sampling_m": 0.000025,
        "mass_inertia_material_joint_modified": False,
        "online_pose_writes_or_thread_constraints_added": False,
        "next_validation": "inspect actual thread clearance and contact-driven rotation/translation in Isaac Sim before assembly claims",
    }
    (output / "geometry_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(Path(__file__).resolve().parents[3], args.output.resolve())
