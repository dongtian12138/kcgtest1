"""Postrun mechanical mating review from preserved native contacts and poses."""
import ast,gzip,json,math,sys
from pathlib import Path
import msgpack,numpy as np
from scipy.spatial.transform import Rotation


def rows_from_native_archive(path):
    needed={'step','phase','object_part_positions_m','object_part_orientations_wxyz'}
    with gzip.open(path,'rb') as stream:
        u=msgpack.Unpacker(stream,raw=False)
        while True:
            try:count=u.read_map_header()
            except msgpack.OutOfData:return
            row={};hand_impulse=0.;clips=set();stop=False;other_hand_points=0
            for _ in range(count):
                key=u.unpack()
                if key in needed:row[key]=u.unpack()
                elif key=='contacts':
                    for _ in range(u.read_map_header()):
                        field=u.unpack()
                        if field!='poll_headers':u.skip();continue
                        for _ in range(u.read_array_header()):
                            paths=None
                            for _ in range(u.read_map_header()):
                                field=u.unpack()
                                if field=='paths':paths=u.unpack()
                                elif field=='contacts':
                                    if paths is None:raise ValueError('native header order is not supported')
                                    side=next((i for i,p in enumerate(paths[:2]) if '/handbase_link' in p),None)
                                    internal=(any('TE_J35FreeSplitPlug' in p for p in paths[:2])
                                        and any('FixedReceptaclePose' in p for p in paths[:2]))
                                    clip_paths={p.split('/Leaf_')[0] for p in paths[2:] if '/Leaf_' in p} if internal else set()
                                    is_stop=internal and sum('SourceMetalStopBox' in p for p in paths[2:])==2
                                    if side is None and not clip_paths and not is_stop:u.skip();continue
                                    positive=0;impulse=0.
                                    for _ in range(u.read_array_header()):
                                        magnitude=0.
                                        for _ in range(u.read_map_header()):
                                            field=u.unpack()
                                            if field=='impulse_n_s':magnitude=math.hypot(*u.unpack())
                                            else:u.skip()
                                        impulse+=magnitude
                                        if magnitude>0:positive+=1
                                    if side is not None:hand_impulse+=impulse
                                    if impulse>1e-10:
                                        clips.update(clip_paths);stop|=is_stop
                                        if side is not None:
                                            if (paths[side].split('/')[-1] not in ('f1Link3','f2Link2','f3Link3')
                                                or not paths[1-side].endswith('/TE_J35FreeSplitPlug/CouplingNut')):
                                                other_hand_points+=positive
                                else:u.skip()
                else:u.skip()
            row.update(hand_impulse_n_s=hand_impulse,source_stop_positive_contact=stop,
                       loaded_clip_count=len(clips),other_hand_positive_points=other_hand_points)
            yield row


def review(directory):
    p=Path(directory);archive=p/'truth_samples.msgpack.gz'
    index=json.loads(Path(str(archive)+'.index.json').read_text())
    installation=json.loads((p/'frozen_model_installation.json').read_text())
    socket=np.asarray(ast.literal_eval(installation['pose_mass_velocity_after'][
        '/World/TEVisualHandoff/FixedReceptaclePose']['world_transform']),float).T
    terminal=next((f for f in [p/'sequence/nut_terminal_release/nut_reindex_controller_result.json',
        p/'socket_transport/nut_terminal_release/nut_reindex_controller_result.json'] if f.exists()),None)
    release=json.loads(terminal.read_text()) if terminal else {}
    end=int(release.get('support_hold_last_step',index['sample_count']))-1
    start=end-1920+1;window=[];first=None;last=None;max_backup=0.;bad_hand=0;count=0;angles=[]
    for row in rows_from_native_archive(archive):
        if row['step']!=count:raise ValueError('discontinuous native observation sequence')
        count+=1;pos=np.asarray(row['object_part_positions_m']);rot=Rotation.from_quat(
            np.asarray(row['object_part_orientations_wxyz'])[:,[1,2,3,0]]).as_matrix();relative=rot[0].T@rot[1]
        axial=float((rot[0].T@(pos[1]-pos[0]))[2]);max_backup=max(max_backup,abs(axial))
        state={'step':row['step'],'phase':row['phase'],'body_depth_mm':-float(socket[:3,2]@(pos[0]-socket[:3,3]))*1000,
            'body_axis_tilt_deg':float(np.degrees(np.arccos(np.clip(-socket[:3,2]@rot[0][:,2],-1,1)))),
            'nut_angle_deg_wrapped':float(np.degrees(np.arctan2(relative[1,0],relative[0,0]))),
            'thrust_coordinate_mm':axial*1000,'hand_impulse_n_s':row['hand_impulse_n_s'],
            'source_stop_positive_contact':row['source_stop_positive_contact'],'loaded_clip_count':row['loaded_clip_count']}
        if first is None:first=state
        last=state;angles.append(state['nut_angle_deg_wrapped'])
        if start<=row['step']<=end:window.append(state)
        if row['phase'].startswith('key_probe_nut'):bad_hand+=row['other_hand_positive_points']
    if count!=index['sample_count']:raise ValueError('archive count differs from the sealed index')
    only_hold=bool(window and all(r['phase']=='nut_index_free_open_hold' for r in window))
    depths=np.asarray([r['body_depth_mm'] for r in window]);angle=np.degrees(np.unwrap(np.radians(angles)))
    checks={'full2s_released_observation':len(window)==1920 and only_hold,
        'all_hand_contacts_absent':bool(window and all(r['hand_impulse_n_s']==0 for r in window)),
        'body_within_original10um_stop_band':bool(window and np.max(abs(depths-14.605))<=.010),
        'original_stop_positive_contact_in_hold':any(r['source_stop_positive_contact'] for r in window),
        'all128_clips_simultaneously_loaded_in_hold':bool(window and max(r['loaded_clip_count'] for r in window)==128),
        'backup_axial_limit_not_used':max_backup<.0005999,'no_unwanted_hand_partner_in_nut_phases':bad_hand==0}
    report={'scope':'POSTRUN_MECHANICAL_MATING_AND_RELEASE_FROM_NATIVE_ARCHIVE',
        'declared_cold_source_diagnosis':(p/'source_stage_probe_result.json').exists(),
        'full_tabletop_visual_success_claimed':False,'online_control_used':False,'sample_count':count,
        'first':first,'final':last,'actual_recorded_relative_nut_rotation_deg':float(angle[-1]-angle[0]),
        'final_release_controller_completed':release.get('completed',False),'release_failure':release.get('failure_reason'),
        'final_window_count':len(window),'final_window_depth_range_mm':[float(depths.min()),float(depths.max())] if len(depths) else None,
        'maximum_absolute_thrust_coordinate_mm':max_backup*1000,'unwanted_hand_partner_positive_points':bad_hand,
        'checks':checks,'mechanical_mating_and_release_verified':all(checks.values()),
        'hand_zero_contact_test_uses_all_raw_impulse_magnitudes_without_threshold':True,
        'source_finger_face_and_key_containment_reviews_still_required':True,
        'per_contact_tensor_friction_is_not_required_or_inferred_for_these_mechanical_checks':True}
    (p/'ending_mating_physical_review.json').write_text(json.dumps(report,indent=2)+'\n');return report


if __name__=='__main__':print(json.dumps(review(sys.argv[1]),indent=2))
