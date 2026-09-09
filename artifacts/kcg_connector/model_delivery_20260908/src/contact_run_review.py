#!/usr/bin/env python3
"""Read-only physical review of a model-delivery run; never emits automatic PASS.

Reads JSONL incrementally, retaining numeric arrays only. No Isaac initialization
or model/scene mutation. Outputs default to this delivery's references/contact_*.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

BODY='/World/TE_J35FreeSplitPlug/Body'
NUT='/World/TE_J35FreeSplitPlug/CouplingNut'


def load_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def vec_stats(a):
    a=np.asarray(a,float)
    if not len(a): return None
    return {'mean':a.mean(0).tolist(),'rms':np.sqrt((a*a).mean(0)).tolist(),
            'maximum_abs':np.abs(a).max(0).tolist()}


def mass_data(run,result):
    if result.get('source_mass_properties'):
        masses=np.array([float(result['source_mass_properties'][p]['physics:mass']) for p in (BODY,NUT)])
        if 'local_com_m' in result:return masses,np.array(result['local_com_m']), 'result.json'
    from pxr import Usd,UsdPhysics
    stage=Usd.Stage.Open(str(run/'connector_candidate.usdc'))
    masses=[];com=[]
    for path in (BODY,NUT):
        api=UsdPhysics.MassAPI(stage.GetPrimAtPath(path))
        masses.append(float(api.GetMassAttr().Get()));com.append(list(api.GetCenterOfMassAttr().Get()))
    return np.array(masses),np.array(com),'READ_ONLY_CANDIDATE_USD'


def raw_geometry_estimate(run,scene,dt):
    """Nearest declared source partition only; never a native collider-ID claim."""
    import trimesh
    band_manifest=load_json(Path(scene['grounding_band_contact_model']['geometry_manifest']))
    seal_manifest=load_json(Path(scene['interfacial_seal_contact_model']['geometry_manifest']))
    entries={
        'rigid_core':seal_manifest['meshes']['rigid_core_without_front_seal'],
        'grounding_band':band_manifest['meshes']['circumferential_band'],
        'front_seal':seal_manifest['meshes']['front_seal_envelope'],
    }
    labels=list(entries);meshes=[]
    for entry in entries.values():
        data=np.load(entry['path'])
        meshes.append(trimesh.Trimesh(vertices=data['vertices_m'],faces=data['faces'],process=False))
    output=[]
    for file in sorted(run.glob('raw_depth_*.npz')):
        raw=np.load(file);count=int(raw['counts'].reshape(-1)[0]);start=int(raw['starts'].reshape(-1)[0])
        sl=slice(start,start+count);p=raw['positions_world_m'][0]
        q=raw['quaternions_wxyz'][0];R=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        points=(raw['points_world_m'][sl]-p)@R
        if not len(points):continue
        distances=np.stack([trimesh.proximity.closest_point(mesh,points)[1] for mesh in meshes],axis=1)
        rank=np.argsort(distances,axis=1);nearest=rank[:,0]
        margin=distances[np.arange(count),rank[:,1]]-distances[np.arange(count),nearest]
        # SDF resolution/contact envelopes imply points near partition boundaries
        # cannot be assigned reliably by nearest source triangle alone.
        ambiguous=(margin<50e-6)|(distances[np.arange(count),nearest]>0.5e-3)
        force=raw['normal_impulse_ns'].reshape(-1)[sl,None]*raw['normals_world'][sl]/dt
        separation=raw['separations_m'].reshape(-1)[sl]
        groups={}
        for i,label in enumerate(labels+['ambiguous']):
            mask=(nearest==i)&~ambiguous if i<len(labels) else ambiguous
            groups[label]={'normal_count':int(mask.sum()),'normal_force_world_n':force[mask].sum(0).tolist(),
                           'normal_load_n':float(np.linalg.norm(force[mask],axis=1).sum()),
                           'maximum_reported_penetration_m':float(np.maximum(-separation[mask],0).max()) if mask.any() else None,
                           'nearest_surface_distance_max_m':float(distances[mask].min(1).max()) if mask.any() else None}
        output.append({'file':str(file),'step':int(raw['step']),'groups':groups,
                       'scope':'BODY_RAW_NORMAL_CONTACT_POINTS_NEAREST_SOURCE_PARTITION_NOT_NATIVE_COLLIDER_ID',
                       'ambiguity_margin_m':50e-6,'maximum_assignment_distance_m':0.5e-3,
                       'friction_separation_available_by_own_shape':False,
                       'limitation':'Nearest partition can misassign deformed contact or closed internal caps. It does not prove seal calibrated or rigid penetration absent. Raw Body points also include pin/clip pairs.'})
    return output


def review(run,geometry=False):
    scene=load_json(run/'assembly_scene.json'); result=load_json(run/'result.json')
    masses,local_com,mass_source=mass_data(run,result)
    path=run/'samples.jsonl';byte_limit=path.stat().st_size
    data={k:[] for k in ('time','phase','step','depth','nut_angle','positions','quats','linear','angular','normal','friction','guide','axial_cap','drive_estimate','penetration')}
    shape_observed=shape_populated=0;filtered_rows=[];parsed=0;incomplete=0
    mapping={'time':'time_s','phase':'phase','step':'step','depth':'body_depth_m','nut_angle':'nut_angle_deg',
             'positions':'positions_world_m','quats':'quaternions_wxyz','linear':'native_linear_velocity_m_s',
             'angular':'native_angular_velocity_rad_s','normal':'normal_wrenches_n_nm','friction':'friction_wrenches_n_nm',
             'guide':'external_guide_enabled','axial_cap':'axial_drive_cap_n','drive_estimate':'drive_spring_only_estimate_nm',
             'penetration':'max_contact_penetration_m'}
    with path.open('rb') as stream:
        while stream.tell()<byte_limit:
            line=stream.readline(byte_limit-stream.tell())
            if not line.endswith(b'\n'):incomplete+=1;break
            row=json.loads(line);parsed+=1
            for key,field in mapping.items():data[key].append(row[field])
            if 'body_shape_contacts' in row:
                shape_observed+=1;shape_populated+=bool(row['body_shape_contacts']['populated_shape_report'])
            if 'filtered_contacts' in row:
                totals={name:np.zeros((2,6)) for name in ('socket_normal','socket_friction','clips_normal','clips_friction')}
                for pair in row['filtered_contacts']:
                    part=pair['sensor_index'];category='socket' if pair['filter_index']==0 else 'clips'
                    totals[category+'_normal'][part]+=pair['normal_wrench_n_nm']
                    totals[category+'_friction'][part]+=pair['friction_wrench_n_nm']
                filtered_rows.append({'step':row['step'],'phase':row['phase'],'depth':row['body_depth_m'],**totals})
    if parsed<2:raise ValueError('At least two complete samples are needed')
    phase=np.array(data.pop('phase'));a={k:np.asarray(v) for k,v in data.items()}
    t=a['time'];sample_dt=np.diff(t);dt=float(np.median(sample_dt))
    R=Rotation.from_quat(a['quats'][:,:,[1,2,3,0]].reshape(-1,4)).as_matrix().reshape(-1,2,3,3)
    com_offset=np.einsum('npij,pj->npi',R,local_com)
    # Native linear velocity is COM velocity. Contact moment in the run is at
    # each prim origin, so shift it before combining with the COM velocity.
    F=a['friction'][:,:,:3];Mcom=a['friction'][:,:,3:]-np.cross(com_offset,F)
    power=(F*a['linear']).sum(2)+(Mcom*a['angular']).sum(2)
    com_world=a['positions']+com_offset
    fd_linear=np.diff(com_world,axis=0)/sample_dt[:,None,None]
    relative_rotation=R[1:]@np.transpose(R[:-1],(0,1,3,2))
    fd_angular=Rotation.from_matrix(relative_rotation.reshape(-1,3,3)).as_rotvec().reshape(-1,2,3)/sample_dt[:,None,None]
    fd_power=(F[1:]*fd_linear).sum(2)+(Mcom[1:]*fd_angular).sum(2)
    total_contact=(a['normal'][:,:,:3]+a['friction'][:,:,:3]).sum(1)
    total_momentum=(a['linear']*masses[None,:,None]).sum(1)
    gravity=np.array([0.,0.,-9.81])*masses.sum()
    acceleration_term=np.diff(total_momentum,axis=0)/sample_dt[:,None]
    residual=total_contact[1:]+gravity-acceleration_term
    socket=np.array(scene['socket_initial_position_world_m'])
    axis=-R[:,0,:,2]  # inserted Body source +Z is approximately world -Z
    tilt=np.rad2deg(np.arccos(np.clip(axis[:,2],-1,1)))
    xy=np.linalg.norm(a['positions'][:,0,:2]-socket[:2],axis=1)
    yaw=np.rad2deg(np.unwrap(np.arctan2(R[:,0,1,0],R[:,0,0,0])))
    try: lead=float(load_json(Path(scene['representative_inner_thread']['manifest_path']))['lead_m'])
    except (KeyError,TypeError):lead=None
    turn=np.flatnonzero(phase=='rotation');depth=a['depth'];nut=a['nut_angle']
    motion={'body_axis_tilt_deg':vec_stats(tilt),'body_xy_axis_offset_m':vec_stats(xy),
            'body_yaw_change_deg':float(yaw[-1]-yaw[0]),'body_yaw_range_deg':float(np.ptp(yaw)),
            'body_initial_final_depth_m':[float(depth[0]),float(depth[-1])],
            'maximum_body_depth_m':float(depth.max()),'nominal_final_depth_m':.014605,
            'final_nominal_depth_error_m':float(depth[-1]-.014605),
            'max_step_depth_change_m':float(abs(np.diff(depth)).max()),
            'max_step_nut_angle_change_deg':float(abs(np.diff(nut)).max()),
            'depth_reversal_travel_m':float(np.maximum(-np.diff(depth),0).sum())}
    if len(turn)>1:
        j,k=turn[[0,-1]];delta_angle=float(nut[k]-nut[j]);delta_depth=float(depth[k]-depth[j])
        windows=[]
        for lo,hi in ((.0055,.0089),(.0089,.0112),(.0112,.0126),(.0126,.0143),(.0143,.01461)):
            idx=turn[(depth[turn]>=lo)&(depth[turn]<hi)]
            if len(idx)>1:
                angle=nut[idx];z=depth[idx]
                span=float(angle.max()-angle.min())
                windows.append({'depth_interval_m':[lo,hi],'sample_count':len(idx),'nut_span_deg':span,
                                'fitted_lead_m_per_revolution':float(np.polyfit(angle,z,1)[0]*360) if span>.1 else None})
        motion['rotation']={'actual_angle_deg':delta_angle,'depth_advance_m':delta_depth,
                            'measured_whole_phase_lead_m_per_revolution':delta_depth/delta_angle*360 if abs(delta_angle)>1e-9 else None,
                            'declared_geometry_lead_m_per_revolution':lead,'depth_windows':windows,
                            'limitation':'Whole-phase lead includes engagement take-up, final seating and recoil; inspect moving windows rather than requiring a single exact slope.'}
    phases={}
    for name in dict.fromkeys(phase):
        idx=np.flatnonzero(phase==name);eligible=idx[idx>0];eligible=eligible[phase[eligible-1]==name]
        w=power[eligible]*sample_dt[eligible-1,None]
        fd_w=fd_power[eligible-1]*sample_dt[eligible-1,None]
        phases[name]={'samples':len(idx),'duration_between_samples_s':float(t[idx[-1]]-t[idx[0]]),
                      'depth_initial_final_m':depth[idx[[0,-1]]].tolist(),
                      'nut_angle_change_deg':float(nut[idx[-1]]-nut[idx[0]]),
                      'maximum_body_axis_tilt_deg':float(tilt[idx].max()),'maximum_body_xy_offset_m':float(xy[idx].max()),
                      'friction_work_body_nut_j':w.sum(0).tolist(),
                      'backward_pose_difference_friction_work_body_nut_j':fd_w.sum(0).tolist(),
                      'native_linear_integral_body_nut_m':(a['linear'][eligible]*sample_dt[eligible-1,None,None]).sum(0).tolist(),
                      'actual_com_displacement_body_nut_m':(com_world[idx[-1]]-com_world[idx[0]]).tolist(),
                      'native_vs_pose_linear_velocity_difference_m_s':vec_stats(a['linear'][eligible]-fd_linear[eligible-1]),
                      'native_vs_pose_angular_velocity_difference_rad_s':vec_stats(a['angular'][eligible]-fd_angular[eligible-1]),
                      'positive_friction_work_body_nut_j':np.maximum(w,0).sum(0).tolist(),
                      'negative_friction_work_body_nut_j':np.minimum(w,0).sum(0).tolist(),
                      'all_external_guide_off':bool((~a['guide'][idx]).all()),
                      'all_axial_drive_cap_zero':bool((a['axial_cap'][idx]==0).all()),
                      'all_reported_rotary_spring_estimate_zero':bool((a['drive_estimate'][idx]==0).all())}
        if name=='free_hold' and len(eligible):
            settled=eligible[t[eligible]>=t[idx[0]]+.25]
            phases[name]['total_force_balance_residual_n']=vec_stats(residual[eligible-1])
            phases[name]['after_0p25s_force_balance_residual_n']=vec_stats(residual[settled-1])
            phases[name]['total_contact_force_mean_n']=total_contact[eligible].mean(0).tolist()
            phases[name]['gravity_force_n']=gravity.tolist()
            phases[name]['body_depth_range_m']=float(np.ptp(depth[idx]))
            phases[name]['body_nut_max_speed_m_s']=np.linalg.norm(a['linear'][idx],axis=2).max(0).tolist()
            phases[name]['body_nut_final_speed_m_s']=np.linalg.norm(a['linear'][idx[-1]],axis=1).tolist()
    filtered={}
    for category in ('socket','clips'):
        for name in dict.fromkeys(phase):
            entries=[v for v in filtered_rows if v['phase']==name]
            if entries:
                loads=np.stack([v[category+'_normal'][:,:3]+v[category+'_friction'][:,:3] for v in entries])
                filtered[f'{name}_{category}_body_nut_force_world_n']={'mean':loads.mean(0).tolist(),'maximum_abs':abs(loads).max(0).tolist()}
    free=phases.get('free_hold');uncertain=[]
    if not result:uncertain.append('Run result.json is absent: this is a bounded live snapshot, not final completion.')
    if not free:uncertain.append('No free-hold samples: release retention is unverified.')
    if shape_populated==0:uncertain.append('Full shape reports contain no populated contacts; cannot certify rigid/seal/band load or rigid penetration by native collider ID.')
    uncertain+=['Finite-step impulse/velocity work is a diagnostic, not native substep energy conservation proof. Native final-step velocities and pose-integrated velocities are reported separately: disagreement prevents an automatic passivity claim.',
                'A USD guide-off flag plus quiet pose is not alone proof of physical detachment; use the combined free-body force balance and trajectory.',
                'Nominal depth alone does not prove correct shell stop or absence of unintended penetration.',
                'This is connector apparatus evidence for the declared model, not mechanical-hand success or exact TE calibration.']
    output={'scope':'READ_ONLY_PHYSICAL_REVIEW_NO_AUTOMATIC_PASS','run':str(run),'complete_result_file_present':bool(result),
            'snapshot_jsonl_bytes':byte_limit,'complete_rows':parsed,'ignored_incomplete_lines':incomplete,
            'physics_dt_median_s':dt,'nonpositive_time_increments':int((sample_dt<=0).sum()),
            'masses_kg':masses.tolist(),'local_com_m':local_com.tolist(),'mass_source':mass_source,
            'motion':motion,'phase_reviews':phases,'filtered_sampled_forces':filtered,
            'shape_report_samples':shape_observed,'populated_shape_report_samples':shape_populated,
            'free_hold_force_balance_definition':'Sum contact forces on Body and Nut + total mass*g - d(sum m*v_COM)/dt; internal joint forces cancel. Residual also includes omitted damping forces and finite-step solver timing.',
            'friction_work_definition':'F_dot_vCOM + (M_origin - (R*localCOM) cross F)_dot_omega_rad_per_s, integrated per step.',
            'contact_penetration_max_body_nut_m':a['penetration'].max(0).tolist(),
            'penetration_scope':'All raw normal contacts include intended compliant overlap; this maximum is not rigid penetration.',
            'unresolved':uncertain,'automatic_pass_or_fail':False}
    if geometry:
        output['raw_depth_partition_estimate']=raw_geometry_estimate(run,scene,dt)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--geometry',action='store_true',help='Offline nearest-source-partition estimates for saved raw depth points, not native shape IDs')
    args=parser.parse_args();run=args.run.resolve()
    output=args.output or Path(__file__).resolve().parents[1]/'references'/f'contact_{run.name}_review.json'
    data=review(run,args.geometry);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'output':str(output),'rows':data['complete_rows'],'final_depth_mm':1000*data['motion']['body_initial_final_depth_m'][1],
                      'has_final_result':data['complete_result_file_present'],'free_hold':data['phase_reviews'].get('free_hold'),
                      'populated_shape_report_samples':data['populated_shape_report_samples']},ensure_ascii=False))


if __name__=='__main__':main()
