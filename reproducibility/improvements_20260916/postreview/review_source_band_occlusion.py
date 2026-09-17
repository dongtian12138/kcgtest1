"""Postrun radial visibility samples of the original fixed red paint geometry.

No SimulationApp, scene edits, controller inputs, or physical execution.
This geometric side-view check complements the actual episode images and
mechanical stop/release review; it does not establish full assembly itself.
"""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import trimesh
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation
from trace_metadata import read_truth_sample, without_cyclic_gc


REPO = Path(__file__).resolve().parents[3]
SOCKET = '/World/TEVisualHandoff/FixedReceptaclePose'
NUT = '/World/TE_J35FreeSplitPlug/CouplingNut'
BAND = SOCKET + '/OfficialVisual/MatingIndicatorRedBand'
NUT_VISUAL = NUT + '/RepresentativeThreadVisual'


def mesh_in_rigid_frame(stage, path, rigid_path):
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsA(UsdGeom.Mesh):
        raise ValueError('Original visible mesh missing: ' + path)
    if str(UsdGeom.Imageable(prim).ComputeVisibility()) == 'invisible':
        raise ValueError('Original visible mesh is hidden: ' + path)
    mesh = UsdGeom.Mesh(prim)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
    if not np.all(counts == 3):
        raise ValueError('This review expects original source triangles')
    vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int).reshape(-1, 3)
    cache = UsdGeom.XformCache()
    rigid = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(rigid_path))).T
    local = np.linalg.inv(rigid) @ np.asarray(cache.GetLocalToWorldTransform(prim)).T
    return vertices @ local[:3, :3].T + local[:3, 3], faces, prim


def review(directory):
    directory = Path(directory).resolve()
    index = json.loads((directory / 'truth_samples.msgpack.gz.index.json').read_text())
    # The archive writer creates this index only in close(). Match its final
    # byte offset as well, rather than require a timing filename that differs
    # between the full runner and the saved-state diagnostic runner.
    archive = directory / 'truth_samples.msgpack.gz'
    if not index['blocks'] or int(index['blocks'][-1]['end']) != archive.stat().st_size:
        raise ValueError('Only a closed archive matching its sealed index can be reviewed')
    last_step = int(index['sample_count']) - 1
    row = without_cyclic_gc(read_truth_sample, directory, last_step)
    model = REPO / 'artifacts/kcg_connector/te_connector_contact_repaired_20260911/connector_model.usdc'
    stage = Usd.Stage.Open(str(model))
    band_v, band_f, band_prim = mesh_in_rigid_frame(stage, BAND, SOCKET)
    nut_v, nut_f, _ = mesh_in_rigid_frame(stage, NUT_VISUAL, NUT)
    if any(p.HasAPI(UsdPhysics.CollisionAPI) or p.HasAPI(UsdPhysics.RigidBodyAPI)
           for p in Usd.PrimRange(band_prim)):
        raise ValueError('Original indicator is no longer paint-only')
    radius = np.linalg.norm(band_v[:, :2], axis=1)
    if np.ptp(radius) > 1e-8:
        raise ValueError('Source band is not the expected fixed cylindrical paint')
    # All source vertices plus every triangle centroid, declared independently
    # of the observed outcome. No point or failed ray is removed.
    points_socket = np.concatenate((band_v, band_v[band_f].mean(axis=1)))
    directions_socket = points_socket.copy()
    directions_socket[:, 2] = 0.
    directions_socket /= np.linalg.norm(directions_socket, axis=1)[:, None]
    install = json.loads((directory / 'frozen_model_installation.json').read_text())
    socket = np.asarray(ast.literal_eval(install['pose_mass_velocity_after'][SOCKET]['world_transform'])).T
    nut_position = np.asarray(row['object_part_positions_m'][1])
    nut_rotation = Rotation.from_quat(np.asarray(row['object_part_orientations_wxyz'][1])[[1, 2, 3, 0]]).as_matrix()
    origins_world = points_socket @ socket[:3, :3].T + socket[:3, 3]
    origins = (origins_world - nut_position) @ nut_rotation
    directions = directions_socket @ socket[:3, :3].T @ nut_rotation
    # Triangle inequality bounds every possible positive ray hit. Filtering
    # by the union segment AABB is an exact broad-phase rejection, not a new
    # approximation to the source surface. It avoids indexing distant threads.
    bound = float(np.max(np.linalg.norm(nut_v, axis=1)) + np.max(np.linalg.norm(origins, axis=1)) + .001)
    endpoints = origins + directions * bound
    lower = np.minimum(origins.min(axis=0), endpoints.min(axis=0)) - 1e-10
    upper = np.maximum(origins.max(axis=0), endpoints.max(axis=0)) + 1e-10
    triangles = nut_v[nut_f]
    keep = np.all(triangles.max(axis=1) >= lower, axis=1) & np.all(triangles.min(axis=1) <= upper, axis=1)
    covered = np.zeros(len(origins), dtype=bool)
    distances = np.full(len(origins), np.nan)
    if np.any(keep):
        mesh = trimesh.Trimesh(vertices=nut_v, faces=nut_f[keep], process=False)
        locations, ray_ids, _ = mesh.ray.intersects_location(origins, directions, multiple_hits=True)
        if len(ray_ids):
            t = np.einsum('ij,ij->i', locations - origins[ray_ids], directions[ray_ids])
            valid = (t > 1e-8) & (t <= bound)
            for rid, distance in zip(ray_ids[valid], t[valid]):
                covered[rid] = True
                if np.isnan(distances[rid]) or distance < distances[rid]:
                    distances[rid] = distance
    output = {
        'scope': 'POSTRUN_ORIGINAL_RED_BAND_RADIAL_SIDE_VIEW_OCCLUSION_SAMPLES',
        'step': last_step, 'phase': row['phase'], 'source_model': str(model),
        'source_band_path': BAND, 'source_occluding_visual_mesh_path': NUT_VISUAL,
        'source_band_radius_m': float(np.median(radius)),
        'source_band_socket_z_range_m': [float(band_v[:, 2].min()), float(band_v[:, 2].max())],
        'source_band_has_physics': False, 'original_mesh_geometry_changed': False,
        'online_control_used': False, 'physics_executed': False,
        'point_selection': 'EVERY_SOURCE_VERTEX_AND_EVERY_SOURCE_TRIANGLE_CENTROID',
        'ray_direction': 'OUTWARD_RADIAL_IN_FIXED_PHYSICAL_SOCKET_FRAME',
        'sample_count': len(origins), 'covered_by_nut_sample_count': int(covered.sum()),
        'all_sampled_radial_views_blocked_by_nut': bool(np.all(covered)),
        'uncovered_sample_indices': np.flatnonzero(~covered).tolist(),
        'positive_hit_distance_range_m': [float(np.nanmin(distances)), float(np.nanmax(distances))] if np.any(covered) else None,
        'source_nut_triangle_count': len(nut_f), 'broad_phase_triangle_count': int(keep.sum()),
        'ray_distance_bound_m': bound,
        'physical_socket_world_from_frame': socket.tolist(),
        'nut_origin_world_m': nut_position.tolist(),
        'sampled_radial_visibility_is_not_all_camera_views_proof': True,
        'actual_episode_image_review_still_required': True,
        'full_assembly_success_claimed': False,
    }
    (directory / 'source_band_radial_occlusion_review.json').write_text(json.dumps(output, indent=2) + '\n')
    return {k: v for k, v in output.items() if k != 'uncovered_sample_indices'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(review(args.directory), indent=2))
