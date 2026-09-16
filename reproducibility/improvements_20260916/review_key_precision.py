"""Separate stored-pose arithmetic from physical key/slot contact residuals."""
import argparse
import ast
import itertools
import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation
from trace_metadata import read_truth_sample, without_cyclic_gc

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
source = args.run.resolve()
index = json.loads((source / 'truth_samples.msgpack.gz.index.json').read_text())
assert (source / 'motion_timing.json').is_file()
assert index['blocks'][-1]['end'] == (source / 'truth_samples.msgpack.gz').stat().st_size
row = without_cyclic_gc(read_truth_sample, source, 228525)
stage = Usd.Stage.Open(str(root / 'artifacts/kcg_connector/te_connector_contact_repaired_20260911/connector_model.usdc'))
dimensions = json.loads((root / 'reproducibility/assembly_20260916/key_backlash_dimensions.json').read_text())
install = json.loads((source / 'frozen_model_installation.json').read_text())
socket_path = '/World/TEVisualHandoff/FixedReceptaclePose'
socket = np.asarray(ast.literal_eval(install['pose_mass_velocity_after'][socket_path]['world_transform']), float).T
key = UsdGeom.Mesh(stage.GetPrimAtPath('/World/TE_J35FreeSplitPlug/Body/SourceGuideKey_002'))
local = np.asarray(key.GetPointsAttr().Get())
transform = np.asarray(UsdGeom.Xformable(key).GetLocalTransformation())
vertices = (np.c_[local, np.ones(len(local))] @ transform)[:, :3]
planes = dimensions['keys'][2]['planes']
position = np.asarray(row['object_part_positions_m'][0])
quat = np.asarray(row['object_part_orientations_wxyz'][0])


def rotation(q, dtype):
    q = np.asarray(q, dtype=dtype)
    w, x, y, z = q / np.sqrt(np.sum(q * q))
    return np.asarray([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                       [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                       [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], dtype=dtype)


def gaps(p, q, dtype):
    S = socket.astype(dtype)
    R = S[:3, :3].T @ rotation(q, dtype)
    t = S[:3, :3].T @ (np.asarray(p, dtype=dtype) - S[:3, 3])
    points = vertices.astype(dtype) @ R.T + t
    inside = points[points[:, 2] < 0]
    return np.asarray([(inside*1000) @ np.asarray(plane['normal'], dtype=dtype)
                       - dtype(plane['d_mm']) for plane in planes], dtype=dtype) * dtype(.001)


original_R = socket[:3, :3].T @ Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
original_p = socket[:3, :3].T @ (position - socket[:3, 3])
points = vertices @ original_R.T + original_p
inside = points[points[:, 2] < 0]
original_gaps = np.asarray([inside*1000 @ np.asarray(plane['normal']) - plane['d_mm'] for plane in planes]) * .001
original = float(original_gaps.min())
assert abs(original - (-2.14765949235185e-6)) < 1e-14
extended = gaps(position, quat, np.longdouble)
assert np.array_equal(position, position.astype(np.float32).astype(float))
# The recorder normalizes its float64 copy of the native float32 quaternion
# while computing the COM, before archiving it. Verify that provenance here.
native_quat = quat.astype(np.float32).astype(float)
assert np.max(np.abs(native_quat/np.linalg.norm(native_quat) - quat)) < 1e-15
values = np.r_[position, quat].astype(np.float32)
lower = (values.astype(float) + np.nextafter(values, np.float32(-np.inf)).astype(float)) / 2
upper = (values.astype(float) + np.nextafter(values, np.float32(np.inf)).astype(float)) / 2
corner_gaps = []
for choices in itertools.product((0, 1), repeat=7):
    perturbed = np.where(choices, upper, lower)
    corner_gaps.append(float(gaps(perturbed[:3], perturbed[3:], np.longdouble).min()))
# Conservative rigid-pose Lipschitz bound; excludes solver, cooking and source
# geometry uncertainty. It is not an estimate of total simulator precision.
half = np.maximum(values.astype(float) - lower, upper - values.astype(float))
translation_bound = max(np.abs(socket[:3, :3] @ np.asarray(plane['normal'])) @ half[:3] for plane in planes)
q_error = float(np.linalg.norm(half[3:]))
rotation_bound = 4*q_error/(np.linalg.norm(quat)-q_error)*float(np.max(np.linalg.norm(vertices, axis=1)))
contacts = [header for header in row['contacts']['poll_headers']
            if any('SourceGuideKey_002' in path for path in header['paths'])]
native_min = min(point['separation_m'] for header in contacts for point in header['contacts'])
socket_prim = stage.GetPrimAtPath(socket_path + '/OfficialVisual/Geometry')
report = {'scope': 'POSTRUN_ORIGINAL_WITNESS_PRECISION_AND_CONTACT_REPRESENTATION',
          'source_run': str(source), 'step': row['step'], 'tolerance_um_unchanged': 2.,
          'physical_or_control_parameters_changed': False,
          'original_geometry_gap_um': original*1e6,
          'extended_precision_geometry_gap_um': float(extended.min()*1e6),
          'arithmetic_difference_um': float((extended.min()-np.longdouble(original))*1e6),
          'body_positions_exact_float32_values': True,
          'body_quaternion_reconstructs_from_normalized_float32': True,
          'body_stored_pose_rounding_cell_corner_gap_range_um': [min(corner_gaps)*1e6, max(corner_gaps)*1e6],
          'body_stored_pose_rounding_conservative_gap_bound_um': (translation_bound+rotation_bound)*1e6,
          'native_report_minimum_separation_um': native_min*1e6,
          'native_key_contact_points': sum(len(header['contacts']) for header in contacts),
          'key_collision_approximation': str(key.GetPrim().GetAttribute('physics:approximation').Get()),
          'socket_collision_approximation': str(socket_prim.GetAttribute('physics:approximation').Get()),
          'key_and_socket_rest_offsets_m': [key.GetPrim().GetAttribute('physxCollision:restOffset').Get(), socket_prim.GetAttribute('physxCollision:restOffset').Get()],
          'original_check_still_failed': original < -2e-6,
          'interpretation': ['Extended arithmetic on the same stored poses cannot erase the observed breach.',
                             'The pose-rounding bounds cover stored Body input quantization only, not all internal PhysX arithmetic or collision-cooking uncertainty.',
                             'The native manifold independently reports penetration, but its contact-generation time need not match the post-step pose time. The nearby six-frame alignment check finds native separation closest to the preceding pose, rather than a geometry mismatch.',
                             'This witness uses a convex key and a static triangle-mesh socket; an SDF-grid explanation does not apply to this pair.',
                             'A solver-iteration comparison is still required; this report alone does not establish that higher iterations solve the complete task.']}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
