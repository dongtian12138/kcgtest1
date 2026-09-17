"""Inspect ended-run control traces and native contacts near key-gap witnesses."""
import argparse, ast, json, math
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src/kcg_connector'),str(ROOT/'src/kcg_connector/isaac')]
from trace_metadata import iter_truth_fields, without_cyclic_gc


def review(run):
    run=Path(run).resolve(); index=json.loads((run/'truth_samples.msgpack.gz.index.json').read_text())
    assert (run/'motion_timing.json').exists() and index['blocks'][-1]['end']==(run/'truth_samples.msgpack.gz').stat().st_size
    turns=[];controls={}
    for path in sorted(run.glob('socket_transport/nut_rotation*/nut_rotation_controller_result.json')):
        result=json.loads(path.read_text());rows=[json.loads(line) for line in (path.parent/'nut_rotation_control_samples.jsonl').read_text().splitlines()]
        controls.update({r['step']:r for r in rows})
        errors=np.asarray([r['pivot_position_tracking_error_m'] for r in rows]);wrench=np.asarray([r['interface_wrench'] for r in rows])
        max_i=int(np.argmax(np.linalg.norm(errors[:,:2],axis=1)))
        turns.append({'stage':path.parent.name,'first_step':result['first_step'],'last_step':result['last_step'],
            'actual_command_deg':abs(result['last_applied_rotation_command_deg']),'elapsed_physics_s':len(rows)/960,
            'maximum_encoder_pivot_xy_error_mm':1000*float(np.linalg.norm(errors[max_i,:2])),
            'maximum_error_step':rows[max_i]['step'],'maximum_error_command_deg':abs(rows[max_i]['commanded_rotation_deg']),
            'final_encoder_pivot_xy_error_mm':1000*float(np.linalg.norm(errors[-1,:2])),
            'maximum_filtered_force_n':float(np.linalg.norm(wrench[:,:3],axis=1).max()),
            'maximum_filtered_bending_nm':float(np.linalg.norm(wrench[:,3:5],axis=1).max()),
            'maximum_filtered_twist_nm':float(np.abs(wrench[:,5]).max()),'early_regrasp':result.get('early_regrasp'),
            'normal_stop_reason':result.get('normal_stop_reason'),'outer_abort':result.get('outer_abort_reason')})
    key_review=json.loads((run/'source_key_containment_review.json').read_text())
    dimensions=json.loads((ROOT/'reproducibility/assembly_20260916/key_backlash_dimensions.json').read_text())
    stage=Usd.Stage.Open(str(ROOT/'artifacts/kcg_connector/te_connector_contact_repaired_20260911/connector_model.usdc'))
    install=json.loads((run/'frozen_model_installation.json').read_text())
    socket=np.asarray(ast.literal_eval(install['pose_mass_velocity_after']['/World/TEVisualHandoff/FixedReceptaclePose']['world_transform']),float).T
    results=[]
    for key_no,witness in enumerate(key_review['minimum_gap_witnesses']):
        if witness['gap_m']>=-2e-6:continue
        mesh=UsdGeom.Mesh(stage.GetPrimAtPath(f'/World/TE_J35FreeSplitPlug/Body/SourceGuideKey_{key_no:03d}'))
        local=np.asarray(mesh.GetPointsAttr().Get());matrix=np.asarray(UsdGeom.Xformable(mesh).GetLocalTransformation());vertices=(np.c_[local,np.ones(len(local))]@matrix)[:,:3]
        planes=dimensions['keys'][key_no]['planes'];center=witness['step'];window=[]
        for row in iter_truth_fields(run,{'phase','object_part_positions_m','object_part_orientations_wxyz','contacts'},center-3,center+3):
            R=socket[:3,:3].T@Rotation.from_quat(np.roll(row['object_part_orientations_wxyz'][0],-1)).as_matrix();pos=socket[:3,:3].T@(np.asarray(row['object_part_positions_m'][0])-socket[:3,3]);pts=vertices@R.T+pos;inside=pts[pts[:,2]<0]
            gap=float(min((inside*1000@np.asarray(p['normal'])-p['d_mm']).min() for p in planes))*.001
            headers=[h for h in row['contacts']['poll_headers'] if any(f'SourceGuideKey_{key_no:03d}' in p for p in h['paths'])];points=[p for h in headers for p in h['contacts']]
            control=controls.get(row['step']);item={'step':row['step'],'phase':row['phase'],'source_key_gap_um':gap*1e6,
                'native_key_contact_minimum_separation_um':min((p['separation_m'] for p in points),default=math.nan)*1e6,
                'native_key_point_count':len(points),'native_key_impulse_norm_sum_n_s':sum(math.hypot(*p['impulse_n_s']) for p in points),
                'body_depth_mm':-pos[2]*1000,'body_lateral_mm':math.hypot(*pos[:2])*1000,'body_axis_tilt_deg':float(np.degrees(np.arccos(np.clip(-R[2,2],-1,1))))}
            if control:item['control_before_step']={'command_deg':abs(control['commanded_rotation_deg']),'encoder_pivot_xy_error_mm':math.hypot(*control['pivot_position_tracking_error_m'][:2])*1000,'filtered_wrench_n_nm':control['interface_wrench'],'load_compensation_wrench_n_nm':control['contact_load_interface_wrench_n_nm']}
            if window:item['native_separation_minus_previous_post_pose_gap_um']=item['native_key_contact_minimum_separation_um']-window[-1]['source_key_gap_um']
            window.append(item)
        results.append({'source_key_number_one_based':key_no+1,'minimum_witness':witness,'window':window})
    return {'scope':'ENDED_EPISODE_CONTROL_AND_KEY_WITNESS_DIAGNOSIS','run':str(run),'online_truth_used':False,'physics_executed':False,'key_band_um_unchanged':2.,'turns':turns,'total_actually_executed_command_deg':sum(t['actual_command_deg'] for t in turns),'turn_physics_seconds':sum(t['elapsed_physics_s'] for t in turns),'key_witness_windows':results,'limitations':['The wrist virtual pivot is a robot/grasp reference, not the observed physical Nut center.','Native contact-generation separation and post-step poses may differ by a physics tick.','Correlation of a key residual with load or wrist error does not prove a causal controller fix.']}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=without_cyclic_gc(review,a.run)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as h:json.dump(result,h,indent=2);h.write('\n')
    print(json.dumps({'output':str(a.output),'turns':result['turns'],'key_witness_count':len(result['key_witness_windows'])},indent=2))
