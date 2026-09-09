"""Reference loading of the actual source robot's open hand on CPU.

The commanded forces and couples are finite laboratory inputs, not contact
truth fed to a controller. The production wrist readout and source inertials
are checked before grasping or insertion is attempted.
"""
from pathlib import Path
import json
import time
import xml.etree.ElementTree as ET
import numpy as np
import yaml


def run_robot_wrist_reference_loads(*,repository,world,robot_data,ft_tree,contact_view,
        hand_paths,recipe,source_sensor,source_metadata,base_run,settings,output):
    from kcg_connector.grasp.carts_v2.models import load_v2_inputs
    import controller
    import omni.usd
    import omni.replicator.core as rep
    from pxr import Gf,Usd,UsdGeom,UsdLux
    from te_foundationpose_handoff_runtime import _author_camera,_camera_cv_pose_from_eye_target
    import cv2
    repository=Path(repository);output=Path(output);started=time.perf_counter()
    (output/'reference_loading_source.py').write_bytes(Path(__file__).read_bytes())
    dt=world.get_physics_dt()
    launch=json.loads(Path(base_run).with_suffix('.launch.json').read_text())['argv']
    config=repository/launch[launch.index('--config')+1]
    model=load_v2_inputs(repository,config_path=config,object_id=source_metadata['object_id']).robot_model
    robot,active,arm_indices,lower,upper,_=robot_data
    arm=np.asarray(source_sensor['active_positions_rad'][:7],float)
    hand=np.asarray(recipe['open_hand_positions_rad'],float)
    hb=next(i for i,p in enumerate(hand_paths) if p.endswith('/handbase_link'))
    row_index=ft_tree._articulation_view._metadata.joint_indices['hand2arm']+1
    inertials=[]
    for link in ET.parse(repository/'src/iiwa_description/urdf/hand.xacro').getroot().findall('link'):
        item=link.find('inertial')
        if item is not None:
            inertials.append((link.get('name'),float(item.find('mass').get('value')),
                              np.fromstring(item.find('origin').get('xyz'),sep=' ')))
    def host(v):
        if hasattr(v,'detach'):return v.detach().cpu().numpy()
        return v.numpy() if hasattr(v,'numpy') else np.asarray(v)
    stage=omni.usd.get_context().get_stage()
    H0=np.asarray(model.forward_kinematics(np.r_[arm,hand],enforce_limits=False)['handbase_link'])
    target=H0[:3,3]+H0[:3,2]*.06
    camera='/World/ReferenceLoadCamera'
    _author_camera(stage,camera,_camera_cv_pose_from_eye_target(target+np.array([.18,-.23,.15]),target),
        resolution=(960,720),focal_length_mm=45.,horizontal_aperture_mm=36.,clipping_range_m=(.01,5.),Gf=Gf,UsdGeom=UsdGeom)
    UsdLux.DomeLight.Define(stage,'/World/ReferenceLoadLight').CreateIntensityAttr(1000.)
    product=rep.create.render_product(camera,(960,720));rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([product.path])
    def capture(name):
        before=host(robot.get_dof_positions(indices=0)).copy();t=float(world.current_time)
        world.pause()
        rep.orchestrator.step(rt_subframes=1,delta_time=0.,pause_timeline=True)
        pixels=np.asarray(rgb.get_data())
        after=host(robot.get_dof_positions(indices=0)).copy()
        audit={'image':f'{name}.png','physics_time_s':t,
               'physics_time_delta_s':float(world.current_time)-t,
               'joint_position_delta_rad':float(np.max(abs(after-before))),
               'image_shape':list(pixels.shape)}
        audit['valid_for_pose_evidence']=bool(pixels.ndim==3 and audit['joint_position_delta_rad']==0 and audit['physics_time_delta_s']==0)
        if pixels.ndim==3:cv2.imwrite(str(output/f'{name}.png'),cv2.cvtColor(pixels[:,:,:3],cv2.COLOR_RGB2BGR))
        (output/f'{name}_capture_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        world.play()
        return audit
    # The centre of the applied force is an explicit point in handbase coordinates.
    lever=np.array([.03,-.02,.01])
    loads=[('Fx_plus',[.5,0,0],[0,0,0]),('Fy_minus',[0,-.5,0],[0,0,0]),
           ('Fz_plus',[0,0,.5],[0,0,0]),('Mx_plus',[0,0,0],[.02,0,0]),
           ('My_minus',[0,0,0],[0,-.02,0]),('Mz_plus',[0,0,0],[0,0,.02])]
    phases=[('unloaded_initial',1.,np.zeros(3),np.zeros(3),'hold')]
    for name,f,t in loads:
        f=np.asarray(f,float);t=np.asarray(t,float)
        phases.extend([(name+'_ramp',.25,f,t,'ramp'),(name+'_hold',.35,f,t,'hold'),
                       (name+'_unload',.25,f,t,'unload'),(name+'_zero',.25,np.zeros(3),np.zeros(3),'hold')])
    phases.append(('unloaded_final',.5,np.zeros(3),np.zeros(3),'hold'))
    samples=[];images=[];count=0;baseline=None
    stream=(output/'reference_samples.jsonl').open('x',buffering=1)
    world.play()
    try:
        for phase,duration,force,couple,mode in phases:
            n=round(duration/dt)
            for i in range(n):
                u=(i+1)/n;s=10*u**3-15*u**4+6*u**5
                factor=s if mode=='ramp' else 1-s if mode=='unload' else 1.
                f=force*factor;t=couple*factor
                target_arm,audit=controller.gravity_biased_arm_target(robot,arm_indices,arm,lower,upper,settings,
                    arm_damping_nm_s_rad=settings['arm_damping'])
                if audit['saturated']:raise RuntimeError('Reference test arm command saturated')
                robot.set_dof_position_targets(np.r_[target_arm,hand][None,:],indices=0,dof_indices=active)
                contact_view.apply_forces_and_torques_at_pos(forces=[f],torques=[t],positions=[lever],indices=[hb],local_frame=True)
                world.step(render=False)
                q=host(robot.get_dof_positions(indices=0)).ravel()[active]
                v=host(robot.get_dof_velocities(indices=0)).ravel()[active]
                fk=model.forward_kinematics(q,enforce_limits=False);H=np.asarray(fk['handbase_link'])
                gravity=np.zeros(6)
                for name,m,com in inertials:
                    T=np.asarray(fk[name]);p=T[:3,:3]@com+T[:3,3];F=np.array([0.,0.,-9.81*m])
                    gravity+=np.r_[F,np.cross(p-H[:3,3],F)]
                raw=host(ft_tree.get_measured_joint_forces())[row_index].copy()
                canonical=np.r_[-H[:3,:3]@raw[:3],-H[:3,:3]@raw[3:]]
                reference=np.r_[H[:3,:3]@f,H[:3,:3]@(t+np.cross(lever,f))]
                measured=canonical-gravity
                normal,_,_,_,counts,starts,_=contact_view.get_raw_contact_data()
                normal=host(normal).ravel();counts=host(counts).ravel();starts=host(starts).ravel()
                normal_load=sum(float(np.abs(normal[int(a):int(a+c)]).sum())/dt for a,c in zip(starts,counts))
                row={'step':count,'time_s':float(world.current_time),'phase':phase,'active_q_rad':q.tolist(),
                    'active_velocity_rad_s':v.tolist(),'raw_sensor_wrench':raw.tolist(),
                    'canonical_world_wrench':canonical.tolist(),'source_gravity_world_wrench':gravity.tolist(),
                    'gravity_subtracted_world_wrench':measured.tolist(),'known_external_world_wrench':reference.tolist(),
                    'applied_force_handbase_n':f.tolist(),'applied_couple_handbase_nm':t.tolist(),
                    'force_application_point_handbase_m':lever.tolist(),'posthoc_hand_normal_load_n':normal_load}
                stream.write(json.dumps(row,separators=(',',':'))+'\n');samples.append(row);count+=1
                if baseline is not None:
                    residual=measured-baseline
                    if np.linalg.norm(residual[:3])>3.0400615 or np.linalg.norm(residual[3:])>.2:
                        raise RuntimeError('Reference loading exceeded independent wrist force/moment stop')
            if phase=='unloaded_initial':
                baseline=np.mean([r['gravity_subtracted_world_wrench'] for r in samples[-round(.25/dt):]],axis=0)
                # Output verification must not become a prerequisite for this
                # independently recorded force measurement sequence.
        images.append(capture('final_open_hand'))
    finally:
        stream.close()
    result={'scope':'ACTUAL_SOURCE_HAND_FREE_SPACE_WRIST_REFERENCE_LOADING_NOT_GRASP_OR_ASSEMBLY',
            'physics_dt_s':dt,'source_robot_and_inertials_preserved':True,'hand_closing_requested':False,
            'source_hand_mass_kg':sum(x[1] for x in inertials),'baseline_gravity_residual_world':baseline.tolist(),
            'contact_truth_used_as_controller_input':False,'known_loads_are_explicit_physics_inputs':True,
            'images':images,'maximum_recorded_hand_normal_load_n':max(r['posthoc_hand_normal_load_n'] for r in samples),
            'cases':[],'wall_seconds':time.perf_counter()-started}
    for name,_,_ in loads:
        rr=[r for r in samples if r['phase']==name+'_hold'][-round(.15/dt):]
        measured=np.array([r['gravity_subtracted_world_wrench'] for r in rr]);expected=np.array([r['known_external_world_wrench'] for r in rr])
        error=measured-baseline-expected
        result['cases'].append({'name':name,'mean_measured_delta_world':(measured-baseline).mean(0).tolist(),
            'mean_reference_world':expected.mean(0).tolist(),'maximum_abs_delta_error':np.max(abs(error),axis=0).tolist()})
    rr=[r for r in samples if r['phase']=='unloaded_final'][-round(.25/dt):]
    result['final_zero_delta_world']=(np.mean([r['gravity_subtracted_world_wrench'] for r in rr],axis=0)-baseline).tolist()
    (output/'reference_result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
