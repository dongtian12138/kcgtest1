"""Read-only helpers reused from the completed independent terminal-release audit."""
import ast
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import msgpack
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from trace_metadata import read_truth_sample, without_cyclic_gc

ROOT = Path(__file__).resolve().parents[3]


def load(path):
    return json.loads(path.read_text())


@contextmanager
def indexed_gzip(path, first_step):
    with path.open('rb') as compressed:
        if first_step:
            index = load(Path(str(path) + '.index.json'))
            block = index['blocks'][first_step // index['block_size']]
            compressed.seek(block['offset'])
        with gzip.GzipFile(fileobj=compressed, mode='rb') as stream:
            yield stream


def selective_rows(path, first_step=0, last_step=None):
    """Skip unused pin arrays, preserving every relevant raw point unthresholded."""
    needed = {'step', 'phase', 'simulation_time_s', 'active_positions_rad',
              'object_part_positions_m', 'object_part_orientations_wxyz',
              'native_robot_link_pose_audit', 'object_part_sleeping'}
    coverage = {'all_provided_native_report_points_retained', 'contact_report_channels_agree',
                'contact_observation_backend', 'physics_step_callback_count'}
    with indexed_gzip(path, first_step) as stream:
        u = msgpack.Unpacker(stream, raw=False)
        while True:
            try:
                count = u.read_map_header()
            except msgpack.OutOfData:
                return
            row = {'headers': [], 'coverage': {}}
            for _ in range(count):
                key = u.unpack()
                if key in needed:
                    row[key] = u.unpack()
                elif key == 'contacts':
                    for _ in range(u.read_map_header()):
                        field = u.unpack()
                        if field in coverage:
                            row['coverage'][field] = u.unpack()
                        elif field != 'poll_headers':
                            u.skip()
                        else:
                            for _ in range(u.read_array_header()):
                                paths = None
                                header = {}
                                for _ in range(u.read_map_header()):
                                    field = u.unpack()
                                    if field == 'paths':
                                        paths = u.unpack()
                                        header['paths'] = paths
                                    elif field == 'contacts':
                                        if paths is None:
                                            raise ValueError('unsupported native header order')
                                        relevant = (any('/handbase_link' in p for p in paths[:2])
                                            or any('/Leaf_' in p or 'SourceMetalStopBox' in p for p in paths[2:]))
                                        if not relevant:
                                            u.skip()
                                            continue
                                        points = []
                                        for _ in range(u.read_array_header()):
                                            point = {}
                                            for _ in range(u.read_map_header()):
                                                f = u.unpack()
                                                if f in ('impulse_n_s', 'position_m'):
                                                    point[f] = u.unpack()
                                                else:
                                                    u.skip()
                                            points.append(point)
                                        header['contacts'] = points
                                    else:
                                        u.skip()
                                if 'contacts' in header:
                                    row['headers'].append(header)
                else:
                    u.skip()
            if last_step is not None and row['step'] >= last_step:
                return
            if row['step'] >= first_step:
                yield row


def summarize_contacts(headers):
    hand = 0.0
    clips = set()
    stops = []
    for h in headers:
        paths = h['paths']
        impulse = sum(math.hypot(*p['impulse_n_s']) for p in h['contacts'])
        if any('/handbase_link' in p for p in paths[:2]):
            hand += impulse
        internal = (any('TE_J35FreeSplitPlug' in p for p in paths[:2])
                    and any('FixedReceptaclePose' in p for p in paths[:2]))
        if internal and impulse > 1e-10:
            clips.update(p.split('/Leaf_')[0] for p in paths[2:] if '/Leaf_' in p)
            if sum('SourceMetalStopBox' in p for p in paths[2:]) == 2:
                stops.append({'paths': paths, 'header_impulse_n_s': impulse})
    return hand, clips, stops


def pose_summary(row, socket):
    p = np.asarray(row['object_part_positions_m'])
    r = Rotation.from_quat(np.asarray(row['object_part_orientations_wxyz'])[:, [1, 2, 3, 0]]).as_matrix()
    return {
        'body_depth_mm': -float(socket[:3, 2] @ (p[0] - socket[:3, 3])) * 1000,
        'body_tilt_deg': float(np.degrees(np.arccos(np.clip(-socket[:3, 2] @ r[0][:, 2], -1, 1)))),
        'thrust_coordinate_mm': float((r[0].T @ (p[1] - p[0]))[2]) * 1000,
    }


def socket_for(run):
    install = load(run / 'frozen_model_installation.json')
    return np.asarray(ast.literal_eval(install['pose_mass_velocity_after'][
        '/World/TEVisualHandoff/FixedReceptaclePose']['world_transform'])).T


def classify_original_surface_groups(groups):
    contract = load(ROOT / 'src/kcg_connector/config/visual_assembly_v1_contact_regions.json')
    geometry = load(ROOT / 'artifacts/grasp_capacity_20260914/selected_two_nail_geometry.json')
    result = {}
    nail_first, nail_end = contract['nail_shell_face_range_zero_based_half_open']
    for name, group in groups.items():
        source = ROOT / f'src/iiwa_description/meshes/hand/{name}.STL'
        assert hashlib.sha256(source.read_bytes()).hexdigest() == contract['bindings'][name]
        mesh = trimesh.load(source, process=False)
        pad = np.load(ROOT / f'artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/TERMINAL_PAD_EXACT_SOURCE_V2/{name}_PAD_BODY_raw_source_local_m.npz')['source_face_indices']
        local = np.asarray([v for _, v in group])
        _, distance, face = trimesh.proximity.closest_point(mesh, local)
        pad_mask = np.isin(face, pad)
        nail_mask = (face >= nail_first) & (face < nail_end)
        allowed = geometry['allowed_contact_regions'][name]
        admitted = (pad_mask.copy() if 'PAD' in allowed else np.zeros(len(group), dtype=bool))
        if 'NAIL' in allowed:
            admitted |= nail_mask
        worst = int(np.argmax(distance))
        result[name] = {
            'positive_contact_points_without_impulse_threshold': len(group),
            'steps_with_positive_points': len(set(s for s, _ in group)),
            'first_positive_step': group[0][0], 'last_positive_step': group[-1][0],
            'allowed_original_regions': allowed, 'pad_points': int(pad_mask.sum()),
            'nail_points': int(nail_mask.sum()), 'disallowed_points': int((~admitted).sum()),
            'maximum_projection_residual_m': float(distance[worst]),
            'maximum_residual_step': group[worst][0],
            'uncertain_projection_points_above_original_500um': int((distance > .0005).sum()),
        }
    return result
