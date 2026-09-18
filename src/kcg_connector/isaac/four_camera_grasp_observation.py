"""One fixed global key frame and a simultaneous palm visibility measurement.

This first probe does not claim the subsequent two-leg route or continuous
tracking is validated. Only images, CAD dimensions and encoder FK are inputs.
"""
from pathlib import Path
from time import perf_counter
import json
import math
import numpy as np


def observe(repository,runtime,stepper,arguments,output,rig):
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf,UsdGeom
    from four_camera_rig import camera_spec,intrinsics
    from te_foundationpose_handoff_runtime import (
        _author_camera,_capture_rgbd,_close_rgbd_resources,_execute_held_plug_path,_json_ready)
    from kcg_connector.te_rgbd_pose_provider import estimate_held_plug_key_from_depth
    from te_plug_five_dof_geometry import estimate_plug_rear_circle_from_float_depth,_visible_face_geometry

    count=runtime.get('body_key_observation_events',0)
    if count:raise RuntimeError('The fixed global key camera may initialize the key only once')
    root=Path(repository);out=Path(output)/'postgrasp_key';out.mkdir(parents=True,exist_ok=False)
    world=runtime['world'];stage=omni.usd.get_context().get_stage()
    q=np.asarray(stepper.latest[0],float)
    hand=np.asarray(runtime['inputs'].robot_model.forward_kinematics(tuple(q),enforce_limits=False)['handbase_link'])
    nominal=np.asarray(runtime['control_plan']['object_from_hand_row_major']).reshape(4,4)
    seed=hand@np.linalg.inv(nominal)
    sample_step=int(stepper.step_index)-1;sample_time=float(world.current_time)
    started=perf_counter();resources={};images={}
    record={'scope':'SINGLE_FIXED_GLOBAL_KEY_AND_PALM_VISIBILITY_PREFIX',
            'robot_sample_step':sample_step,'physics_time_s':sample_time,
            'world_from_hand_encoder':hand.tolist(),'active_positions_rad':q.tolist(),
            'online_object_or_contact_truth_used':False,'full_assembly_claimed':False}
    # Rendering produces a synchronized sensor sample at sample_time. Before
    # consuming the result, the same held-body controller advances physical
    # time by the measured observation latency (see below).
    world.pause()
    try:
        for role in ('global_2','palm'):
            spec,pose=camera_spec(root,rig,role,hand)
            _author_camera(stage,spec['prim_path'],pose,resolution=tuple(spec['resolution_px']),
                focal_length_mm=spec['focal_length_mm'],horizontal_aperture_mm=spec['horizontal_aperture_mm'],
                clipping_range_m=tuple(spec['clipping_range_m']),Gf=Gf,UsdGeom=UsdGeom)
            world.render()
            folder=out/role
            capture=_capture_rgbd(rep=rep,resources=resources,camera_path=spec['prim_path'],
                resolution=tuple(spec['resolution_px']),output_dir=folder,warmup_frames=3,rt_subframes=4)
            images[role]=(spec,pose,np.load(folder/'depth_m.npy'),capture)
        spec,camera,depth,capture=images['global_2'];K=intrinsics(spec)
        key=estimate_held_plug_key_from_depth(depth,K,camera,seed)
        runtime['body_key_observation_events']=1
        record.update(capture=capture,intrinsics_3x3=K.tolist(),world_from_camera_cv=camera.tolist(),
                      key_measurement=key,key_observation_event_count=1)
        if not key.get('key_direction_measured'):
            raise RuntimeError('Fixed global camera2 did not resolve the body key')
        body=np.asarray(key['world_from_plug_row_major']).reshape(4,4)
        record['hand_from_body_visual_memory']=(np.linalg.inv(hand)@body).tolist()
        spec,camera,depth,capture=images['palm'];K=intrinsics(spec)
        cad=root/'artifacts/kcg_connector/vision/sam6d_segmentation_run19_observation_v1/D38999_26FJ35PN_VISUAL.obj'
        radius,offset=_visible_face_geometry(cad)
        expected=np.linalg.inv(camera)@np.r_[body[:3,3]-offset*body[:3,2],1.]
        if expected[2]<=0:raise RuntimeError('Visible rear face is behind the declared palm camera')
        uv=(K@expected[:3])[:2]/expected[2]
        y,x=np.indices(depth.shape)
        radius_px=1.5*float(K[0,0])*radius/expected[2]
        mask=((x+.5-uv[0])**2+(y+.5-uv[1])**2<=radius_px**2)
        mask &= np.isfinite(depth)&(depth>0)&(np.abs(depth-expected[2])<.012)
        import cv2
        cv2.imwrite(str(out/'palm/visual_roi.png'),mask.astype(np.uint8)*255)
        measured=estimate_plug_rear_circle_from_float_depth(depth_m=depth,mask=mask,
            intrinsics=K,mesh_path=cad,pixel_center_offset_px=.5,plane_iterations=128)
        palm_body=camera@np.asarray(measured['camera_from_object'])
        record['palm_observation']={'capture':capture,'world_from_camera_cv':camera.tolist(),
            'intrinsics_3x3':K.tolist(),'world_from_plug_five_dof':palm_body.tolist(),
            'metrics':measured['metrics'],'axial_yaw_measured':False,
            'roi_source':'SIMULTANEOUS_GLOBAL2_KEY_IMAGE_AND_ORIGINAL_CAD_REAR_FACE_DIMENSIONS',
            'online_object_or_contact_truth_used':False}
    except Exception as error:
        record.update(observation_failure=str(error))
    finally:
        _close_rgbd_resources(resources)
    latency=perf_counter()-started
    if float(world.current_time)!=sample_time:
        raise RuntimeError('Synchronized image acquisition unexpectedly advanced physics')
    dt=float(arguments.dynamic_settings['physics_dt_s'])
    # This causal serial schedule is equivalent to processing an old image
    # while the robot keeps holding: its result is unavailable to control
    # until all intervening physics and force-feedback samples have run.
    steps=math.ceil(latency/dt)
    held=np.asarray(runtime['nail_body_ft_auditor'].samples[-1]['active_targets_rad'][:7])
    probe={'authorization':{'simulation_only':True,'hardware_authorized':False},
           'motion':{'maximum_transport_joint_speed_rad_s':.15}}
    advance=_execute_held_plug_path(world,stepper,runtime['nail_body_ft_auditor'],runtime['grasp_result'],
        arguments.dynamic_settings,np.repeat(held[None,:],steps,axis=0),probe,
        phase='key_probe_observation_latency_hold')
    record['observation_latency']={'capture_and_processing_wall_s':latency,
        'physics_hold_duration_s':steps*dt,'physics_hold_steps':steps,
        'availability_physics_time_s':float(world.current_time),'availability_step':int(stepper.step_index),
        'measurement_was_not_used_during_latency_hold':True,'hold_execution':advance}
    (out/'camera_and_estimate.json').write_text(json.dumps(_json_ready(record),indent=2)+'\n')
    if record.get('observation_failure'):raise RuntimeError(record['observation_failure'])
    if not advance['completed']:raise RuntimeError('Original held-body limits stopped the observation latency hold')
    runtime['fixed_camera_grasp_observation']=record
    return record
