"""Offline final release coverage supplement; run only after physical recording ends.

Reads the indexed archived final release interval, never writes to the run.
The per-frame observations supplement existing any/max checks; they do not
change acceptance rules or establish a complete tabletop episode by themselves.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from verify_sealed_final_evidence import (
    load, selective_rows, summarize_contacts, pose_summary, socket_for,
    classify_original_surface_groups, without_cyclic_gc,
)


def review(run):
    run = Path(run).resolve()
    candidates = [run / 'socket_transport/nut_terminal_release/nut_reindex_controller_result.json',
                  run / 'sequence/nut_terminal_release/nut_reindex_controller_result.json']
    release_path = next((p for p in candidates if p.exists()), None)
    if release_path is None:
        raise ValueError('No recorded terminal release')
    release = load(release_path)
    index = load(run / 'truth_samples.msgpack.gz.index.json')
    if (not release.get('completed') or release.get('outer_abort_reason') is not None
            or release['last_step'] != index['sample_count']):
        raise ValueError('Requires the completed final release and its complete matching archive')
    first = release['first_step']; opening = release['opening_last_step']; end = release['support_hold_last_step']
    socket = socket_for(run)
    dt = load(run / 'trace_metadata.json')['physics_dt_s']
    hold = []; actors = Counter(); phases = Counter(); coverage = Counter(); groups = defaultdict(list)
    unclassified = []; unexpected = []; maximum_axial = 0.0; count = 0
    for row in selective_rows(run / 'truth_samples.msgpack.gz', first, end):
        assert row['step'] == first + count
        count += 1; step = row['step']; phases[row['phase']] += 1
        pose = pose_summary(row, socket)
        maximum_axial = max(maximum_axial, abs(pose['thrust_coordinate_mm']))
        hand, clips, stops = summarize_contacts(row['headers'])
        for k in ('all_provided_native_report_points_retained', 'contact_report_channels_agree'):
            coverage[k + '_true'] += row['coverage'].get(k) is True
        coverage['one_physics_callback'] += row['coverage'].get('physics_step_callback_count') == 1
        if step >= opening:
            hold.append({'step': step, 'phase': row['phase'], 'hand_impulse_n_s': hand,
                'loaded_clip_set': tuple(sorted(clips)), 'stop_count': len(stops),
                'sleeping': row['object_part_sleeping'], **pose})
            continue
        for header in row['headers']:
            paths = header['paths']; side = next((i for i,p in enumerate(paths[:2]) if '/handbase_link' in p), None)
            if side is None:
                continue
            name = paths[side].split('/')[-1]; partner = paths[1-side]
            native_pose = row['native_robot_link_pose_audit']['poses'].get(name)
            for point in header['contacts']:
                magnitude = math.hypot(*point['impulse_n_s'])
                if magnitude == 0:
                    continue
                actors[(paths[side], partner)] += 1
                if name not in ('f1Link3', 'f2Link2', 'f3Link3') or not partner.endswith('/TE_J35FreeSplitPlug/CouplingNut'):
                    unexpected.append({'step': step, 'paths': paths, 'impulse_n_s': magnitude})
                if native_pose is None or name not in ('f1Link3', 'f2Link2', 'f3Link3'):
                    unclassified.append({'step': step, 'paths': paths, 'impulse_n_s': magnitude})
                    continue
                r = Rotation.from_quat(np.asarray(native_pose['orientation_world_wxyz'])[[1,2,3,0]]).as_matrix()
                groups[name].append((step, (np.asarray(point['position_m'])-native_pose['position_world_m']) @ r))
    assert count == end-first and len(hold) == end-opening
    if not hold:
        raise ValueError('No archived open hold')
    per_link = classify_original_surface_groups(groups)
    sets = {r['loaded_clip_set'] for r in hold}
    return {
        'scope': 'POSTRUN_TERMINAL_RELEASE_COVERAGE_SUPPLEMENT_ONLY', 'run': str(run),
        'original_acceptance_rules_changed': False, 'complete_tabletop_success_claimed': False,
        'reviewed_raw_range_half_open': [first,end], 'sample_count_reviewed': count,
        'phase_counts': dict(phases), 'record_coverage': dict(coverage),
        'hold': {'raw_range_half_open': [opening,end], 'count':len(hold), 'physics_dt_s':dt,
            'duration_from_physical_intervals_s':len(hold)*dt,
            'all_phase_free_open_hold':all(r['phase']=='nut_index_free_open_hold' for r in hold),
            'all_hand_impulse_norm_sums_exactly_zero':all(r['hand_impulse_n_s']==0 for r in hold),
            'depth_range_mm':[min(r['body_depth_mm'] for r in hold),max(r['body_depth_mm'] for r in hold)],
            'source_stop_positive_any_frame':any(r['stop_count']>0 for r in hold),
            'maximum_simultaneously_loaded_clip_count':max(len(r['loaded_clip_set']) for r in hold),
            'source_stop_positive_every_frame':all(r['stop_count']>0 for r in hold),
            'stop_count_range':[min(r['stop_count'] for r in hold),max(r['stop_count'] for r in hold)],
            'loaded_clip_count_range':[min(len(r['loaded_clip_set']) for r in hold),max(len(r['loaded_clip_set']) for r in hold)],
            'stop_and128_simultaneous_every_frame':all(r['stop_count']>0 and len(r['loaded_clip_set'])==128 for r in hold),
            'distinct_loaded_clip_sets':len(sets),
            'sleeping_state_counts':dict(Counter(str(r['sleeping']) for r in hold))},
        'maximum_absolute_thrust_coordinate_mm_during_terminal_release_only':maximum_axial,
        'backup_threshold_mm_unchanged':.5999,
        'unload_and_open': {'raw_range_half_open':[first,opening], 'frames_reviewed':opening-first,
            'actor_pairs':[{'hand':k[0],'partner':k[1],'positive_points':v} for k,v in actors.items()],
            'unexpected_actor_positive_contacts':unexpected, 'unclassified_positive_contacts':unclassified,
            'source_surface_projection_limit_m_unchanged':.0005, 'per_link':per_link},
        'limitations': ['Only final release interval is read; earlier stages and whole-run backup clearance require their own existing reviews.',
            'Missing/zero stop records alone are not treated as loss of contact; sleeping and native-report coverage are reported for interpretation.',
            'No video, red-band visibility, lift or earlier support claim is made.'],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.run.resolve()):
        raise ValueError('Write this independent supplement outside the source run')
    result = without_cyclic_gc(review, args.run)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({'output':str(args.output.resolve()), 'hold':result['hold'],
                      'unload_and_open':result['unload_and_open']}, indent=2))
