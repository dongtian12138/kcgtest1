#!/usr/bin/env python3
"""Partition the source circumferential bulge for a future local contact model.

Offline geometry only. The exterior union, source keys and original assets are
preserved. This does not assign a spring law or install a new runtime collider.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh

from build_te_free_split_plug import _close_hidden_planar_boundaries


def cleaned(mesh):
    vertices, inverse = np.unique(mesh.vertices, axis=0, return_inverse=True)
    faces = inverse[mesh.faces]
    keep = ((faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2]))
    result = trimesh.Trimesh(vertices, faces[keep], process=False)
    if not result.is_volume or np.any(result.area_faces <= 0):
        raise ValueError("partition is not a closed positive nondegenerate volume")
    return result


def prepare(repository, output):
    output.mkdir(parents=True, exist_ok=False)
    source = repository / "artifacts/carts_grasp/CARTS_GRASP_V1/object_models/te_j35/plug_body_visual_mesh.npz"
    data = np.load(source)
    vertices, faces, caps = _close_hidden_planar_boundaries(data["vertices_m"], data["faces"])
    original = trimesh.Trimesh(vertices, faces, process=False)
    if not original.is_volume:
        raise ValueError("the existing source body closure is not a positive volume")
    # Place the internal split inside the 17.4117 mm CAD neck, avoiding
    # coincidence with its tessellated surface. The exterior union is unchanged;
    # this is not a supplier spring thickness or a change to the spring outline.
    base_radius = .01735
    # The preceding exact key-root plane created a collinear boolean face.
    # Put this interior partition 4.6 micrometres behind the source key roots;
    # all original key vertices must remain on the rigid partition below.
    z_min, z_max = -.0141, -.00765
    annulus = trimesh.creation.annulus(r_min=base_radius, r_max=.021,
                                      height=z_max-z_min, sections=140)
    annulus.apply_translation([0, 0, (z_min+z_max)/2])
    band = cleaned(trimesh.boolean.intersection([original, annulus], engine="manifold"))
    rigid = cleaned(trimesh.boolean.difference([original, annulus], engine="manifold"))
    union = cleaned(trimesh.boolean.union([rigid, band], engine="manifold"))
    volume_error = abs(rigid.volume + band.volume-original.volume)
    if volume_error > original.volume * 1e-6:
        raise ValueError("partition volume does not conserve the original body volume")
    # Deterministic original-surface samples, including every key vertex.
    raw, triangles = data["vertices_m"], data["faces"]
    slab = (raw[:, 2] >= -.007646) & (raw[:, 2] <= -.000761)
    outside = slab & (np.linalg.norm(raw[:, :2], axis=1) > .0183)
    key_faces = np.all(slab[triangles], axis=1) & np.any(outside[triangles], axis=1)
    keys = raw[np.unique(triangles[key_faces])]
    samples = np.vstack((raw[np.linspace(0, len(raw)-1, 2000).astype(int)], keys))
    _, union_distance, _ = trimesh.proximity.closest_point(union, samples)
    _, key_distance, _ = trimesh.proximity.closest_point(rigid, keys)
    if union_distance.max() > 5e-8 or key_distance.max() > 5e-8:
        raise ValueError("source exterior/key samples were displaced by the partition")
    meshes = {}
    for name, mesh in (("rigid_body", rigid), ("circumferential_band", band)):
        path = output / (name + ".npz")
        np.savez_compressed(path, vertices_m=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces, np.int32))
        meshes[name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            vertex_count=len(mesh.vertices), triangle_count=len(mesh.faces),
                            volume_m3=float(mesh.volume), bounds_m=mesh.bounds.tolist())
    result = dict(
        scope="OFFLINE_GEOMETRY_PARTITION_FOR_LOCAL_COMPLIANCE_DEVELOPMENT_ONLY",
        source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        original_asset_modified=False, installed_in_runtime=False,
        mass_inertia_joint_or_friction_changed=False, spring_law_assigned=False,
        component_identity="Circumferential bulge is consistent with the manufacturer's grounding-finger location; source STEP supplies a continuous simplified surface.",
        internal_split_surface="Representative cylindrical interior partition; not a supplier spring thickness or number of fingers.",
        internal_base_radius_m=base_radius, axial_partition_m=[z_min, z_max],
        source_rear_neck_radius_m=.0174117,
        original_hidden_caps=caps, original_volume_m3=float(original.volume),
        partition_volume_error_m3=float(volume_error),
        original_surface_sample_count=len(samples),
        maximum_sample_to_partition_union_distance_m=float(union_distance.max()),
        original_key_face_count=int(key_faces.sum()),
        maximum_key_vertex_to_rigid_partition_distance_m=float(key_distance.max()),
        meshes=meshes,
        unresolved=["effective stiffness and damping", "physical spring thickness/cuts", "contact-model resolution dependence", "other missing pin socket and seal details"],
        physical_assembly_or_compliance_validated=False)
    (output / "geometry_manifest.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(Path(__file__).resolve().parents[3], parser.parse_args().output.resolve())
