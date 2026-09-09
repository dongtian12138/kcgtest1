"""One bounded grasp/torque experiment against an explicit laboratory Nut mount."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml


def run_mounted_nut_torque_probe(*,repository,world,robot_data,ft_tree,contact_view,
        hand_paths,recipe,fixture_pose,source_sensor,source_metadata,base_run,settings,output):
    from kcg_connector.grasp.carts_v2.models import load_v2_inputs
    import controller

    repository=Path(repository);output=Path(output);dt=world.get_physics_dt()
    launch=json.loads(Path(base_run).with_suffix(".launch.json").read_text())["argv"]
    config_path=repository/launch[launch.index("--config")+1]
    document=yaml.safe_load(config_path.read_text());dynamic=document["dynamic"]
    model=load_v2_inputs(repository,config_path=config_path,object_id=source_metadata["object_id"]).robot_model
    robot,active,arm_indices,lower,upper,_=robot_data
    arm=np.asarray(source_sensor["active_positions_rad"][:7],float)
    open_hand=np.asarray(recipe["open_hand_positions_rad"],float)
    contact_hand=np.asarray(recipe["first_contact_hand_positions_rad"],float)
    close_hand=contact_hand.copy();close_hand[1:]+=.03
    lo,hi=robot.get_dof_limits(indices=0,dof_indices=active)
    hand_lo,hand_hi=lo.numpy()[0,7:],hi.numpy()[0,7:]
    if np.any(close_hand>hand_hi) or np.any(open_hand<hand_lo):
        raise ValueError("fixture grasp targets exceed original joint limits")
    desired=np.asarray(recipe["nominal_closing_effort_reference_nm"],float)
    if desired.shape!=(3,) or np.any(desired<=0) or np.any(desired>=.9):
        raise ValueError("finite sub-stop closing references required")
    load_probe=recipe.get("actuator_load_probe")
    normal_feedback=recipe.get("normal_force_feedback")
    fixed_preload=recipe.get('fixed_preload_targets_rad')
    fixed_reaction_stop=float(recipe.get('fixed_preload_reaction_stop_nm',.9))
    wrist_force_reference=float(recipe.get('wrist_force_reference_n',3.0400615))
    if not 3.0400615<=wrist_force_reference<=5.:raise ValueError('bounded mounted wrist force reference required')
    fixed_caps=None
    wrench_filter_tau=float(recipe.get('contact_wrench_filter_time_constant_s',0.))
    hold_fraction=recipe.get('torque_hold_fraction_of_stop')
    torque_hold_s=float(recipe.get('torque_hold_duration_s',.5))
    axial_probe=recipe.get('axial_stiffness_probe')
    axial_reference_m=None
    settling=recipe.get('fixed_grip_settling')
    settling_result=None
    if settling is not None and (fixed_preload is None or not 1.<=float(settling['maximum_s'])<=20.
            or float(settling['window_s'])!=1. or not 0<float(settling['position_drift_m'])<=2e-6
            or not 0<float(settling['force_drift_n'])<=.02 or not 0<float(settling['torque_drift_nm'])<=.005):
        raise ValueError('fixed-grip settling requires the bounded measured-state criterion')
    if axial_probe is not None and (fixed_preload is None or load_probe or normal_feedback
            or not (0<float(axial_probe['displacement_m'])<=20e-6
                    or (float(axial_probe['displacement_m'])==0. and axial_probe.get('stationary_control') is True))
            or float(axial_probe['ramp_s'])<1. or not 0<float(axial_probe['hold_s'])<=1.):
        raise ValueError('axial stiffness probe requires the fixed grip and a bounded20micrometre displacement')
    if (not 0<=wrench_filter_tau<=.05 or (wrench_filter_tau>0 and fixed_preload is None)
            or (hold_fraction is not None and (fixed_preload is None or not 0<float(hold_fraction)<=.9 or not 0<torque_hold_s<=1.))):
        raise ValueError('mounted fixed-grip filter/hold must remain within the declared main-controller bandwidth and finite limits')
    if fixed_preload is not None:
        fixed_preload=np.asarray(fixed_preload,float)
        fixed_caps=np.asarray(recipe['fixed_preload_drive_caps_nm'],float)
        fixed_bounds=np.asarray(recipe['fixed_preload_bounds_rad'],float)
        if (load_probe or normal_feedback or fixed_preload.shape!=(4,) or fixed_caps.shape!=(4,)
                or fixed_bounds.shape!=(2,4) or not np.isfinite(np.r_[fixed_preload,fixed_caps,fixed_bounds.ravel()]).all()
                or np.any(fixed_preload<fixed_bounds[0]) or np.any(fixed_preload>fixed_bounds[1])
                or np.any(fixed_preload<hand_lo) or np.any(fixed_preload>hand_hi)
                or not np.array_equal(fixed_caps,np.array([1.,2.,2.,2.])) or fixed_reaction_stop!=2.):
            raise ValueError('fixed-preload laboratory mode must retain the current bounded hand configuration')
    observer=None;last_force_estimate=None;normal_reference=None
    actuation_target=None;last_applied_actuation=np.zeros(3);old_reaction_crossings=0
    if load_probe:
        load_interface=load_probe.get("interface","native_effort")
        load_target=np.asarray(load_probe["target_closing_motor_torque_nm"],float)
        command_cap=float(load_probe["command_limit_nm"])
        reaction_stop=float(load_probe["projected_reaction_stop_nm"])
        velocity_stop=float(load_probe["native_joint_velocity_stop_rad_s"])
        velocity_damping=float(load_probe["viscous_joint_damping_nm_s_rad"])
        ramp_duration=float(load_probe["ramp_duration_s"])
        if (load_target.shape!=(3,) or not np.isfinite(load_target).all() or np.any(load_target<=0)
                or np.max(load_target)>command_cap or not 0<command_cap<=(2.7 if normal_feedback else 2.35)
                or not .9<=reaction_stop<=2.0 or not 0<velocity_stop<=.18
                or velocity_damping!=2. or ramp_duration!=3.
                or max(load_probe["normal_target_n"])>14. or load_probe["normal_reference_n"]!=15.
                or load_interface not in ("native_effort","native_pd_drive")):
            raise ValueError("source-sized finite laboratory actuation envelope differs")
        motor_caps=np.full(3,command_cap)
    if normal_feedback:
        from te_three_finger_wrench_observer import ThreeFingerWrenchObserver
        if not load_probe or load_interface!="native_pd_drive":
            raise ValueError("normal feedback requires the retained native PD drive")
        normal_targets=np.asarray(normal_feedback["target_normal_force_n"],float)
        motor_caps=np.asarray(normal_feedback["motor_caps_nm"],float)
        effort_levers=np.asarray(normal_feedback["source_coupled_normal_effort_levers_m"],float)
        normal_stop=float(normal_feedback["estimated_normal_stop_n"])
        condition_stop=float(normal_feedback["maximum_normalized_condition"])
        if (normal_targets.shape!=(3,) or motor_caps.shape!=(3,) or effort_levers.shape!=(3,)
                or not np.isfinite(np.r_[normal_targets,motor_caps,effort_levers]).all()
                or np.any(normal_targets<=0) or np.max(normal_targets)>13.
                or np.any(motor_caps<=0) or np.max(motor_caps)>2.7
                or np.any(motor_caps>command_cap) or np.any(effort_levers<=0) or np.max(effort_levers)>.2
                or normal_stop!=13.5 or condition_stop!=100.
                or normal_feedback["empirical_underestimate_margin_n"]!=1.5
                or normal_feedback["regulation_time_constant_s"]!=1/6):
            raise ValueError("the declared geometry/sensor normal-load envelope differs")
        observer=ThreeFingerWrenchObserver(repository,model,normal_feedback["source_geometry_plan"])
    torque_stop=float(recipe["diagnostic_torque_stop_nm"])
    angle=float(recipe["diagnostic_wrist_rotation_deg"])*np.pi/180
    if not 0<torque_stop<=1.6 or not 0<angle<=np.deg2rad(8):
        raise ValueError("the fixture torque and rotation bounds are 1.6 Nm and 8 degrees")
    if not lower[6]<arm[6]+angle<upper[6]:raise ValueError("original wrist joint limit would be exceeded")
    H0=np.asarray(model.forward_kinematics(np.r_[arm,open_hand],enforce_limits=False)["handbase_link"])
    test=arm.copy();test[6]+=angle
    H1=np.asarray(model.forward_kinematics(np.r_[test,open_hand],enforce_limits=False)["handbase_link"])
    R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
    if np.linalg.norm(H1[:3,3]-H0[:3,3])>1e-5 or not np.allclose(H1[:3,:3],H0[:3,:3]@R,atol=1e-5):
        raise ValueError("the source last wrist axis is not the declared fixture axis")
    inertials=[]
    for link in ET.parse(repository/"src/iiwa_description/urdf/hand.xacro").getroot().findall("link"):
        i=link.find("inertial")
        if i is not None:inertials.append((link.get("name"),float(i.find("mass").get("value")),
                                          np.fromstring(i.find("origin").get("xyz"),sep=" ")))
    row_index=ft_tree._articulation_view._metadata.joint_indices["hand2arm"]+1
    fixture_origin=fixture_pose[:3,3];axis=fixture_pose[:3,2]
    bias=np.zeros(6);tare=np.zeros(4);hand=open_hand.copy();phase="initial_open_hold"
    stop=None;count=0;max_torque=0.;max_loading_torque=0.;max_force=0.;max_pad=np.zeros(3);samples=[]
    filtered_fixture_wrench=None;latest_control_torque=0.;maximum_control_torque=0.;hold_completed=False;latest_hand_world=None
    raw_stream=(output/"mounted_grasp_samples.jsonl").open("x",buffering=1)
    def host(v):return v.detach().cpu().numpy() if hasattr(v,"detach") else v.numpy() if hasattr(v,"numpy") else np.asarray(v)
    if fixed_caps is not None:
        robot.set_dof_max_efforts(fixed_caps[None,:],indices=0,dof_indices=active[7:])
        if not np.allclose(host(robot.get_dof_max_efforts(indices=0,dof_indices=active[7:])),fixed_caps[None,:],atol=1e-6,rtol=0):
            raise RuntimeError('fixed-preload native caps differ')
    def native_channels():
        pp,qq=contact_view.get_world_poses();vv,ww=contact_view.get_velocities()
        return {"joint_position":host(robot.get_dof_positions(indices=0)).copy(),
            "joint_velocity":host(robot.get_dof_velocities(indices=0)).copy(),
            "joint_projected_force":host(robot.get_dof_projected_joint_forces(indices=0)).copy(),
            "wrist_force":host(ft_tree.get_measured_joint_forces()).copy(),
            "rigid_position":host(pp).copy(),"rigid_orientation":host(qq).copy(),
            "rigid_linear_velocity":host(vv).copy(),"rigid_angular_velocity":host(ww).copy()}
    image_records=[];image_resources=None
    fabric_output_each_step=bool(recipe.get("update_fabric_each_physics_step",False))
    refresh_articulation_before_render=bool(recipe.get("refresh_articulation_kinematics_before_render",False))
    render_refresh_state_violation=False
    fabric_settings={};maximum_output_sync_native_delta=0.
    if recipe.get("record_lab_stills"):
        import omni.usd
        import omni.replicator.core as rep
        from pxr import Gf,UsdGeom,UsdLux
        from te_foundationpose_handoff_runtime import _author_camera,_camera_cv_pose_from_eye_target
        stage=omni.usd.get_context().get_stage();camera_path="/World/MountedHandLabEvidenceCamera"
        _author_camera(stage,camera_path,_camera_cv_pose_from_eye_target(
            fixture_origin+[.13,-.16,.13],fixture_origin+[0,0,.015]),
            resolution=(800,600),focal_length_mm=32.,horizontal_aperture_mm=36.,clipping_range_m=(.01,5.),Gf=Gf,UsdGeom=UsdGeom)
        UsdLux.DomeLight.Define(stage,"/World/MountedHandLabEvidenceLight").CreateIntensityAttr(1000.)
        product=rep.create.render_product(camera_path,(800,600));rgb=rep.AnnotatorRegistry.get_annotator("rgb");rgb.attach([product.path])
        depth=rep.AnnotatorRegistry.get_annotator("distance_to_image_plane");depth.attach([product.path])
        image_resources=(rep,product,rgb,depth)
        import carb
        from isaacsim.core.simulation_manager import SimulationManager
        fabric_settings['fabric_active']=bool(SimulationManager.is_fabric_enabled())
        if SimulationManager.is_fabric_enabled():
            import omni.physxfabric.bindings._physxFabric as fabric_bindings
            for name in ("SETTING_FABRIC_ENABLED","SETTING_FABRIC_USE_GPU_INTEROP",
                         "SETTING_FABRIC_UPDATE_TRANSFORMATIONS","SETTING_FABRIC_UPDATE_JOINT_STATES",
                         "SETTING_FABRIC_UPDATE_VELOCITIES"):
                key=getattr(fabric_bindings,name,None)
                fabric_settings[name]={"path":key,"value":carb.settings.get_settings().get(key) if key else None}
        for key in ("/physics/suppressReadback","/app/useFabricSceneDelegate"):
            fabric_settings[key]={"path":key,"value":carb.settings.get_settings().get(key)}
    def capture_still(label):
        nonlocal render_refresh_state_violation
        if image_resources is None:return
        if render_refresh_state_violation:
            raise RuntimeError("render refresh previously changed native state; no further refresh attempted")
        import carb,cv2,omni.kit.app,omni.timeline
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.experimental.utils.backend import use_backend
        channels=native_channels
        def state():return np.concatenate([v.ravel() for v in channels().values()])
        before=state().copy();time_before=float(world.current_time);was_playing=bool(world.is_playing())
        refresh_deltas=None
        if refresh_articulation_before_render:
            # This is the missing GPU-articulation prefix in the installed
            # SimulationContext.render. Test it separately from app stepping.
            prior=channels()
            world.physics_sim_view.update_articulations_kinematic()
            following=channels()
            refresh_deltas={k:float(np.max(abs(following[k]-v))) for k,v in prior.items()}
            (output/(label+"_kinematics_refresh_readback.json")).write_text(json.dumps({
                "world_time_s":time_before,"native_channel_max_abs_changes":refresh_deltas,
                "before":{k:v.tolist() for k,v in prior.items()},
                "after":{k:v.tolist() for k,v in following.items()}},indent=2)+"\n")
            if max(refresh_deltas.values())>0 or world.current_time!=time_before:
                render_refresh_state_violation=True
                raise RuntimeError("GPU articulation render refresh changed native state: "+str(refresh_deltas))
        if SimulationManager.is_fabric_enabled() and (not fabric_output_each_step or refresh_articulation_before_render):
            from omni.physxfabric import get_physx_fabric_interface
            get_physx_fabric_interface().force_update(dt,time_before)
        def render_pose_readback():
            # This reads/updates the rendering hierarchy cache, never a physics
            # pose. Native-state invariance is checked across the whole capture.
            record={}
            for backend in ("fabric","usd"):
                try:
                    with use_backend(backend,raise_on_unsupported=True,raise_on_fallback=True):
                        p,q=contact_view.get_world_poses()
                        record[backend]={"position_world_m":host(p).copy().tolist(),
                                         "orientation_world_wxyz":host(q).copy().tolist()}
                except Exception as error:record[backend]={"read_error":str(error)}
            return record
        hierarchy_before=render_pose_readback()
        timeline=omni.timeline.get_timeline_interface();auto=timeline.is_auto_updating()
        settings_car=carb.settings.get_settings();play=settings_car.get("/app/player/playSimulations")
        rep,_,rgb,depth=image_resources
        try:
            timeline.set_auto_update(False);timeline.commit_silently();settings_car.set("/app/player/playSimulations",False)
            for _ in range(3):omni.kit.app.get_app().update()
            hierarchy_before_render=render_pose_readback()
            settings_car.set("/app/player/playSimulations",play)
            rep.orchestrator.step(rt_subframes=1,delta_time=0.,pause_timeline=not was_playing)
        finally:
            settings_car.set("/app/player/playSimulations",play);timeline.set_auto_update(auto);timeline.commit_silently()
        hierarchy_after=render_pose_readback()
        delta=float(np.max(abs(state()-before)));rgba=np.asarray(rgb.get_data());depth_image=np.asarray(depth.get_data())
        if delta>0 or world.current_time!=time_before or rgba.ndim!=3 or not rgba.size or depth_image.shape!=(600,800):
            raise RuntimeError("mounted-hand evidence capture changed native state or returned no image")
        cv2.imwrite(str(output/(label+".png")),cv2.cvtColor(rgba[:,:,:3],cv2.COLOR_RGB2BGR))
        np.save(output/(label+"_depth_m.npy"),depth_image)
        camera=UsdGeom.Camera.Get(stage,camera_path)
        from pxr import Usd
        camera_usd=np.asarray(UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        image_records.append({"image":label+".png","rendered_depth":label+"_depth_m.npy",
            "step_after":count-1,"time_s":time_before,"native_state_max_abs_delta":delta,
            "render_pose_readback_before":hierarchy_before,
            "render_pose_readback_before_orchestrator":hierarchy_before_render,
            "render_pose_readback_after":hierarchy_after,
            "world_from_camera_cv":(camera_usd@np.diag([1.,-1.,-1.,1.])).tolist(),
            "camera_focal_length":float(camera.GetFocalLengthAttr().Get()),
            "camera_horizontal_aperture":float(camera.GetHorizontalApertureAttr().Get()),
            "camera_vertical_aperture":float(camera.GetVerticalApertureAttr().Get()),
            "articulation_kinematics_refresh_deltas":refresh_deltas,
            "render_geometry_matches_native_physics_verified":False})
    def step(arm_target,hand_target):
        nonlocal count,stop,max_torque,max_loading_torque,max_force,max_pad,last_applied_actuation,old_reaction_crossings,last_force_estimate,maximum_output_sync_native_delta
        nonlocal filtered_fixture_wrench,latest_control_torque,maximum_control_torque,latest_hand_world
        drive_arm,audit=controller.gravity_biased_arm_target(robot,arm_indices,arm_target,lower,upper,
                                                           settings,arm_damping_nm_s_rad=settings["arm_damping"])
        if audit["saturated"]:stop="ARM_COMMAND_SATURATION";return None
        targets=np.r_[drive_arm,hand_target]
        if actuation_target is not None:
            velocity=host(robot.get_dof_velocities(indices=0,dof_indices=active[8:])).ravel()
            if load_interface=="native_effort":
                # Explicit-effort diagnostic retained for failure reproduction.
                last_applied_actuation=np.clip(actuation_target-velocity_damping*velocity,-command_cap,command_cap)
                robot.set_dof_efforts(last_applied_actuation[None,:],indices=0,dof_indices=active[8:])
            else:
                # Keep the original native K/D drive. Its position offset is
                # selected using encoders and the known stiffness; the actual
                # motor effort remains limited by the native drive itself.
                position=host(robot.get_dof_positions(indices=0,dof_indices=active[8:])).ravel()
                if normal_feedback:
                    if last_force_estimate is None or normal_reference is None:
                        stop="MISSING_ROBOT_SIDE_NORMAL_FORCE_ESTIMATE";return None
                    # Reuse the already established 1/6 s finger-loop time
                    # constant. Feedback is a force estimate from robot-side
                    # sensors, never a value from the contact recorder below.
                    correction=dt/(1/6+dt)*effort_levers*(normal_reference-last_force_estimate["normal_force_n"])/12.
                else:
                    correction=position+actuation_target/12.-hand_target[1:]
                hand_target[1:]=np.clip(hand_target[1:]+np.clip(correction,-.18*dt,.18*dt),
                                       open_hand[1:],preload_upper[1:])
                targets[8:]=hand_target[1:]
                last_applied_actuation=np.clip(12.*(hand_target[1:]-position)-2.*velocity,-motor_caps,motor_caps)
        robot.set_dof_position_targets(targets[None,:],indices=0,dof_indices=active)
        step_time_before=float(world.current_time)
        world.step(render=bool(recipe.get("standard_render_steps",False)))
        step_elapsed=float(world.current_time)-step_time_before
        if abs(step_elapsed-dt)>1e-8:
            stop=f"SDK_RENDER_STEP_PHYSICS_DELTA_DIFFERS: {step_elapsed} versus {dt}"
            raise RuntimeError(stop)
        all_q=robot.get_dof_positions(indices=0).numpy()[0];all_v=robot.get_dof_velocities(indices=0).numpy()[0]
        all_e=robot.get_dof_projected_joint_forces(indices=0).numpy()[0]
        q=all_q[active];v=all_v[active];e=all_e[active]
        fk=model.forward_kinematics(q,enforce_limits=False);H=np.asarray(fk["handbase_link"])
        latest_hand_world=H.copy()
        gravity=np.zeros(6)
        for name,mass,com in inertials:
            T=np.asarray(fk[name]);p=T[:3,:3]@com+T[:3,3];F=np.array([0.,0.,-9.81*mass])
            gravity+=np.r_[F,np.cross(p-H[:3,3],F)]
        raw=host(ft_tree.get_measured_joint_forces())[row_index].copy()
        canonical=np.r_[-H[:3,:3]@raw[:3],-H[:3,:3]@raw[3:]]
        measured=canonical-gravity-bias
        at_fixture=np.r_[measured[:3],measured[3:]+np.cross(H[:3,3]-fixture_origin,measured[:3])]
        raw_torque=float(abs(at_fixture[3:]@axis));raw_force=float(np.linalg.norm(measured[:3]))
        if filtered_fixture_wrench is None:filtered_fixture_wrench=at_fixture.copy()
        else:filtered_fixture_wrench+=dt/(wrench_filter_tau+dt)*(at_fixture-filtered_fixture_wrench)
        feedback=filtered_fixture_wrench if wrench_filter_tau>0 else at_fixture
        torque=float(abs(feedback[3:]@axis));force=float(np.linalg.norm(feedback[:3]))
        latest_control_torque=torque;maximum_control_torque=max(maximum_control_torque,torque)
        if observer is not None and observer.tare_reaction is not None:
            try:last_force_estimate=observer.estimate(q,e[8:],measured,fk=fk)
            except ValueError as error:
                last_force_estimate=None;stop="INVALID_ROBOT_SIDE_FORCE_OBSERVER: "+str(error)
        # The following contact channels are record-only; neither targets nor
        # stopping decisions consume simulator contact points or force truth.
        f,points,normals,sep,counts,starts,ids=contact_view.get_raw_contact_data()
        f=f.numpy().ravel();points=points.numpy();normals=normals.numpy();sep=sep.numpy().ravel()
        record_contacts=[];normal_by_tip=np.zeros(3);truth=np.zeros(6)
        for index,(n,start) in enumerate(zip(counts.numpy().ravel(),starts.numpy().ravel())):
            start,n=int(start),int(n)
            partners=(contact_view.get_actor_paths_from_ids(ids[start:start+n].to("cpu")) if n else [])
            for j in range(start,start+n):
                F=f[j]*normals[j]/dt;truth+=np.r_[F,np.cross(points[j]-H[:3,3],F)]
                link=hand_paths[index].rsplit("/",1)[-1]
                if link in ("f1Link3","f2Link2","f3Link3"):
                    normal_by_tip[("f1Link3","f2Link2","f3Link3").index(link)]+=np.linalg.norm(F)
                record_contacts.append({"sensor_link":link,"other_actor":partners[j-start],"point_world_m":points[j].tolist(),
                    "force_world_n":F.tolist(),"separation_m":float(sep[j])})
        friction,fp,fc,fs=contact_view.get_friction_data();friction=friction.numpy()/dt;fp=fp.numpy()
        friction_rows=[]
        for index,(n,start) in enumerate(zip(fc.numpy().ravel(),fs.numpy().ravel())):
            start,n=int(start),int(n)
            for j in range(start,start+n):
                F=friction[j];truth+=np.r_[F,np.cross(fp[j]-H[:3,3],F)]
                friction_rows.append({"sensor_index":index,"point_world_m":fp[j].tolist(),"force_world_n":F.tolist()})
        native_positions,native_orientations=contact_view.get_world_poses()
        output_sync_delta=0.
        if fabric_output_each_step:
            # Official manual-stepping sequence: publish the completed PhysX
            # result to Fabric after each physics step. This is output only;
            # no body/joint pose setter or extra physics step is used.
            from omni.physxfabric import get_physx_fabric_interface
            before=np.r_[all_q,all_v,all_e,raw,
                         host(native_positions).ravel(),host(native_orientations).ravel()]
            get_physx_fabric_interface().update(dt,float(world.current_time))
            after=np.concatenate([host(robot.get_dof_positions(indices=0)).ravel(),
                host(robot.get_dof_velocities(indices=0)).ravel(),
                host(robot.get_dof_projected_joint_forces(indices=0)).ravel(),
                host(ft_tree.get_measured_joint_forces())[row_index].ravel(),
                *[host(x).ravel() for x in contact_view.get_world_poses()]])
            output_sync_delta=float(np.max(abs(after-before)))
            maximum_output_sync_native_delta=max(maximum_output_sync_native_delta,output_sync_delta)
            if output_sync_delta>0:stop="FABRIC_OUTPUT_SYNCHRONIZATION_CHANGED_NATIVE_STATE"
        native_poses={path.rsplit("/",1)[-1]:{"position_world_m":native_positions.numpy()[i].tolist(),
            "orientation_world_wxyz":native_orientations.numpy()[i].tolist()} for i,path in enumerate(hand_paths)}
        record={"step":count,"phase":phase,"time_s":float(world.current_time),"active_q_rad":q.tolist(),
            "active_velocity_rad_s":v.tolist(),"projected_efforts_nm":e.tolist(),"drive_targets_rad":targets.tolist(),
            "raw_wrist_n_nm":raw.tolist(),"canonical_world_n_nm":canonical.tolist(),"modeled_gravity_world_n_nm":gravity.tolist(),
            "estimated_contact_world_n_nm":measured.tolist(),"estimated_at_fixture_n_nm":at_fixture.tolist(),
            "estimated_axis_torque_magnitude_nm":raw_torque,"estimated_force_norm_n":raw_force,
            "feedback_wrench_at_fixed_fixture_n_nm":feedback.tolist(),
            "control_axis_torque_magnitude_nm":torque,"control_force_norm_n":force,
            "axial_stiffness_probe_reference_m":axial_reference_m,
            "truth_contact_world_n_nm":truth.tolist(),"record_only_normal_load_by_tip_n":normal_by_tip.tolist(),
            "record_only_normal_contacts":record_contacts,"record_only_friction_contacts":friction_rows,
            "record_only_native_hand_link_poses":native_poses,
            "closing_actuation_mode":load_interface if actuation_target is not None else "ORIGINAL_POSITION_DRIVE",
            "closing_native_effort_command_nm":last_applied_actuation.tolist() if actuation_target is not None and load_interface=="native_effort" else None,
            "closing_pd_law_pre_step_effort_estimate_nm":last_applied_actuation.tolist() if actuation_target is not None and load_interface=="native_pd_drive" else None,
            "closing_motor_load_target_nm":actuation_target.tolist() if actuation_target is not None and not normal_feedback else None,
            "normal_force_reference_n":normal_reference.tolist() if normal_reference is not None else None,
            "robot_side_normal_force_estimate_n":last_force_estimate["normal_force_n"].tolist() if last_force_estimate is not None else None,
            "force_observer_normalized_condition":last_force_estimate["normalized_condition"] if last_force_estimate is not None else None,
            "force_observer_uses_object_or_contact_truth":False,
            "physics_to_fabric_output_update_each_step":fabric_output_each_step,
            "output_update_native_state_max_abs_delta":output_sync_delta,
            "actual_world_step_elapsed_s":step_elapsed,
            "world_from_hand_encoder":H.tolist()}
        raw_stream.write(json.dumps(record,separators=(",",":"))+"\n");count+=1
        max_torque=max(max_torque,raw_torque);max_force=max(max_force,raw_force);max_pad=np.maximum(max_pad,normal_by_tip)
        if phase=="torque_loading":max_loading_torque=max(max_loading_torque,raw_torque)
        old_reaction_crossings+=int(np.max(np.abs(e[8:]))>.9)
        if normal_feedback and actuation_target is not None and last_force_estimate is None:
            stop=stop or "MISSING_ROBOT_SIDE_FORCE_OBSERVER"
        elif normal_feedback and actuation_target is not None and last_force_estimate["normalized_condition"]>condition_stop:
            stop="FORCE_OBSERVER_OUTSIDE_DECLARED_CONDITION_RANGE"
        elif normal_feedback and actuation_target is not None and np.any(last_force_estimate["normal_force_n"]>normal_stop):
            stop="ESTIMATED_FINGER_NORMAL_OPERATION_REFERENCE"
        elif normal_feedback and actuation_target is not None and np.any(last_force_estimate["normal_force_n"]<=0.):
            stop="NONPOSITIVE_CONTACT_ESTIMATE_AFTER_GRIP_CONFIRMATION"
        elif actuation_target is not None and (np.any(q[7:]<hand_lo) or np.any(q[7:]>hand_hi)):
            stop="SOURCE_HAND_POSITION_LIMIT"
        elif actuation_target is not None and np.max(np.abs(v[8:]))>velocity_stop:
            stop="SOURCE_CLOSING_VELOCITY_REFERENCE"
        elif fixed_preload is not None and (np.any(q[7:]<hand_lo) or np.any(q[7:]>hand_hi)):
            stop='SOURCE_HAND_POSITION_LIMIT'
        elif fixed_preload is not None and np.max(np.abs(v[8:]))>.18:
            stop='SOURCE_CLOSING_VELOCITY_REFERENCE'
        elif np.max(np.abs(e[8:]))>(fixed_reaction_stop if fixed_preload is not None else reaction_stop if actuation_target is not None else .9):stop="HAND_PROJECTED_EFFORT_REFERENCE"
        elif phase!="initial_open_hold" and force>wrist_force_reference:stop="WRIST_RESULTANT_FORCE_REFERENCE"
        elif phase in ('torque_loading','controlled_torque_hold','axial_stiffness_ramp','axial_stiffness_hold') and torque>=torque_stop:stop="DECLARED_TORQUE_TEST_BOUND_REACHED"
        samples.append([count,torque,force,*normal_by_tip])
        return q,v,e,canonical-gravity
    latest=None;success=False
    try:
        world.play()
        if recipe.get("explicit_fabric_stage_attachment",False):
            import omni.usd,omni.physics.core
            from pxr import UsdUtils
            from omni.physx import get_physx_simulation_interface
            from omni.physxfabric import get_physx_fabric_interface
            stage=omni.usd.get_context().get_stage()
            stage_id=int(UsdUtils.StageCache.Get().GetId(stage).ToLongInt())
            if stage_id<=0:raise RuntimeError("current simulation stage has no valid cache ID")
            readback={"usd_context_stage_id":int(omni.usd.get_context().get_stage_id()),
                "usd_cache_stage_id":stage_id,
                "unified_physics_stage_id":int(omni.physics.core.get_physics_simulation_interface().get_attached_stage()),
                "physx_stage_id":int(get_physx_simulation_interface().get_attached_stage()),
                "robot_tensor_valid":bool(robot.is_physics_tensor_entity_valid()),
                "rigid_tensor_valid":bool(contact_view.is_physics_tensor_entity_valid()),
                "operation":"ATTACH_OUTPUT_INTERFACE_ONLY_NO_PHYSICS_REATTACH"}
            (output/"fabric_stage_attachment_readback.json").write_text(json.dumps(readback,indent=2)+"\n")
            if (readback["usd_context_stage_id"]!=stage_id
                    or any(readback[k] not in (0,stage_id) for k in ("unified_physics_stage_id","physx_stage_id"))
                    or not readback["robot_tensor_valid"] or not readback["rigid_tensor_valid"]):
                raise RuntimeError("active USD and physics stage IDs differ; output attachment not attempted")
            prior=native_channels();time_before=float(world.current_time)
            result=get_physx_fabric_interface().attach_stage(stage_id)
            following=native_channels()
            changes={k:float(np.max(abs(following[k]-v))) for k,v in prior.items()}
            readback.update(returned_value=str(result),native_channel_max_abs_changes=changes,
                            world_time_change_s=float(world.current_time)-time_before,
                            unified_physics_stage_after=int(omni.physics.core.get_physics_simulation_interface().get_attached_stage()),
                            physx_stage_after=int(get_physx_simulation_interface().get_attached_stage()))
            (output/"fabric_stage_attachment_readback.json").write_text(json.dumps(readback,indent=2)+"\n")
            if (max(changes.values())>0 or world.current_time!=time_before
                    or readback["unified_physics_stage_after"]!=readback["unified_physics_stage_id"]
                    or readback["physx_stage_after"]!=readback["physx_stage_id"]):
                render_refresh_state_violation=True
                raise RuntimeError("output stage attachment changed native state")
        warm=[];efforts=[];warm_encoders=[]
        for _ in range(round(.75/dt)):
            latest=step(arm,hand)
            if stop:raise RuntimeError(stop)
            warm.append(latest[3]);efforts.append(latest[2][7:]);warm_encoders.append(latest[0].copy())
        bias=np.mean(warm[-round(.25/dt):],axis=0);tare=np.mean(efforts[-round(.25/dt):],axis=0)
        if observer is not None:
            observer.calibrate_free_space(warm_encoders[-round(.25/dt):],
                np.asarray(efforts)[-round(.25/dt):,1:])
        contact=controller.ParallelEffortContactController(open_hand,close_hand,
            effort_rise_nm=float(dynamic["contact_effort_rise_nm"]),position_error_rad=float(dynamic["contact_position_error_rad"]),
            velocity_absolute_max_rad_s=dynamic.get("contact_velocity_absolute_max_rad_s"),
            consecutive_samples=int(dynamic["contact_consecutive_samples"]),
            endpoint_timeout_samples=round(float(dynamic["contact_endpoint_timeout_s"])/dt),
            hand_stiffness=12.,finger_order=(1,2,3))
        phase="finite_parallel_closure";increment=.18*dt
        for _ in range(round(6./dt)):
            hand=contact.step(latest[0][7:],latest[2][7:]-tare,increment,measured_velocity=latest[1][7:])
            latest=step(arm,hand)
            if stop:raise RuntimeError(stop)
            if contact.failed:raise RuntimeError(contact.failure_reason)
            if contact.complete:break
        if not contact.complete:raise RuntimeError("finite contact confirmation did not complete")
        preload_upper=np.minimum(hand_hi,contact.target+.20);hand=contact.target.copy()
        phase="bounded_preload"
        if fixed_preload is None:
            for _ in range(round(2./dt)):
                measured=latest[2][8:]-tare[1:]
                hand[1:]=np.clip(hand[1:]+np.clip(dt/(1/6+dt)*(desired-measured)/12.,-increment,increment),open_hand[1:],preload_upper[1:])
                latest=step(arm,hand)
                if stop:raise RuntimeError(stop)
        else:
            start_target=hand.copy();ramp_s=max(2.,1.875*np.max(abs(fixed_preload-start_target))/.18)
            for i in range(round(ramp_s/dt)):
                u=(i+1)/round(ramp_s/dt);blend=10*u**3-15*u**4+6*u**5
                hand=start_target+blend*(fixed_preload-start_target);latest=step(arm,hand)
                if stop:raise RuntimeError(stop)
            hand=fixed_preload.copy()
            phase='fixed_preload_hold'
            for _ in range(round(.5/dt)):
                latest=step(arm,hand)
                if stop:raise RuntimeError(stop)
        capture_still("original_preload")
        if settling is not None:
            from collections import deque
            phase='measured_fixed_grip_settling';history=deque(maxlen=round(float(settling['window_s'])/dt));stable=0;checks=[]
            for i in range(round(float(settling['maximum_s'])/dt)):
                latest=step(arm,hand)
                if stop:raise RuntimeError(stop)
                history.append(np.r_[latest_hand_world[:3,3],filtered_fixture_wrench])
                if len(history)==history.maxlen and i%max(1,round(.5/dt))==0:
                    h=np.asarray(history);half=len(h)//2;change=h[half:].mean(0)-h[:half].mean(0)
                    check=dict(elapsed_s=(i+1)*dt,position_drift_m=float(np.linalg.norm(change[:3])),
                        force_drift_n=float(np.linalg.norm(change[3:6])),torque_drift_nm=float(np.linalg.norm(change[6:])))
                    accepted=(check['position_drift_m']<=float(settling['position_drift_m'])
                              and check['force_drift_n']<=float(settling['force_drift_n'])
                              and check['torque_drift_nm']<=float(settling['torque_drift_nm']))
                    check['accepted']=accepted;checks.append(check);stable=stable+1 if accepted else 0
                    if stable>=2:break
            settling_result={'settings':settling,'checks':checks,'accepted':stable>=2}
            if stable<2:raise RuntimeError('fixed grip did not reach the declared measured steady baseline')
        if load_probe:
            # Start at the previous PD law's effort, then perform one planned
            # source-model load ramp. This is confined to the declared Nut
            # fixture; no connector body force or pose setter is introduced.
            actuation_target=np.clip(12.*(hand[1:]-latest[0][8:])
                -(2.*latest[1][8:] if load_interface=="native_effort" else 0.),-1.,1.)
            initial_effort=actuation_target.copy()
            if load_interface=="native_effort":
                robot.set_dof_gains(np.zeros((1,3)),np.zeros((1,3)),indices=0,dof_indices=active[8:])
            else:
                robot.set_dof_max_efforts(motor_caps[None,:],indices=0,dof_indices=active[8:])
                actual_caps=host(robot.get_dof_max_efforts(indices=0,dof_indices=active[8:])).ravel()
                kp,kd=robot.get_dof_gains(indices=0,dof_indices=active[8:])
                if (not np.allclose(actual_caps,motor_caps,atol=1e-6,rtol=0)
                        or not np.allclose(host(kp),12.,atol=1e-6,rtol=0)
                        or not np.allclose(host(kd),2.,atol=1e-6,rtol=0)):
                    raise RuntimeError("native PD caps/gains differ from the declared load configuration")
            if normal_feedback:
                if last_force_estimate is None:raise RuntimeError("no valid normal estimate at handover")
                normal_reference=last_force_estimate["normal_force_n"].copy()
                initial_normal_reference=normal_reference.copy()
            phase="source_sized_load_handover"
            for _ in range(round(.5/dt)):
                latest=step(arm,hand)
                if stop:raise RuntimeError(stop)
            phase="source_sized_effort_ramp"
            for i in range(round(ramp_duration/dt)):
                u=(i+1)*dt/ramp_duration;blend=10*u**3-15*u**4+6*u**5
                actuation_target=initial_effort+blend*(load_target-initial_effort)
                if normal_feedback:normal_reference=initial_normal_reference+blend*(normal_targets-initial_normal_reference)
                latest=step(arm,hand)
                if stop:raise RuntimeError(stop)
            phase="source_sized_effort_hold"
            for _ in range(round(.5/dt)):
                latest=step(arm,hand)
                if stop:raise RuntimeError(stop)
            capture_still("source_sized_load")
        if axial_probe is not None:
            jac=np.asarray(model.geometric_jacobian('handbase_link',tuple(latest[0])))[:,:7]
            weighted=np.vstack((jac[:3],.05*jac[3:]))
            dq=weighted.T@np.linalg.solve(weighted@weighted.T+.0005**2*np.eye(6),np.r_[axis*float(axial_probe['displacement_m']),np.zeros(3)])
            previous=0.
            for level in (-1.,0.,1.,0.):
                phase='axial_stiffness_ramp'
                for i in range(round(float(axial_probe['ramp_s'])/dt)):
                    u=(i+1)/round(float(axial_probe['ramp_s'])/dt);blend=10*u**3-15*u**4+6*u**5
                    fraction=previous+blend*(level-previous);axial_reference_m=fraction*float(axial_probe['displacement_m'])
                    target=arm+fraction*dq;latest=step(target,hand)
                    if stop:raise RuntimeError(stop)
                phase='axial_stiffness_hold';axial_reference_m=level*float(axial_probe['displacement_m'])
                for _ in range(round(float(axial_probe['hold_s'])/dt)):
                    latest=step(target,hand)
                    if stop:raise RuntimeError(stop)
                previous=level
            stop='COMPLETED_BOUNDED_AXIAL_STIFFNESS_PROBE'
        else:
            phase="torque_loading";duration=max(1.875*abs(angle)/np.deg2rad(2.),
                np.sqrt((10/np.sqrt(3))*abs(angle)/np.deg2rad(.3284481531389871)))
            for i in range(round(duration/dt)):
                u=min(1.,(i+1)*dt/duration);theta=angle*(10*u**3-15*u**4+6*u**5)
                target=arm.copy();target[6]+=theta
                if actuation_target is None and fixed_preload is None:
                    measured=latest[2][8:]-tare[1:]
                    hand[1:]=np.clip(hand[1:]+np.clip(dt/(1/6+dt)*(desired-measured)/12.,-increment,increment),open_hand[1:],preload_upper[1:])
                latest=step(target,hand)
                if stop:break
                if hold_fraction is not None and latest_control_torque>=float(hold_fraction)*torque_stop:
                    phase='controlled_torque_hold'
                    for _ in range(round(torque_hold_s/dt)):
                        latest=step(target,hand)
                        if stop:break
                    hold_completed=stop is None
                    if hold_completed:stop='COMPLETED_BOUNDED_TORQUE_HOLD'
                    break
        success=True
    except Exception as error:
        if stop is None:stop=str(error)
    finally:
        if count:
            try:capture_still("final_lab_state")
            except Exception as error:
                (output/"image_capture_error.txt").write_text(str(error)+"\n")
        world.pause();raw_stream.close()
        result={"scope":"EXPLICIT_FIXED_NUT_GRASP_TORQUE_DIAGNOSTIC_NOT_ASSEMBLY",
            "lab_recipe":recipe,"fixture_pose_world":fixture_pose.tolist(),"completed_loading_attempt":success,
            "stop_reason":stop,"sample_count":count,"maximum_measured_axis_torque_nm":max_torque,
            "maximum_measured_loading_axis_torque_nm":max_loading_torque,
            "maximum_measured_wrist_force_n":max_force,"record_only_maximum_normal_load_by_tip_n":max_pad.tolist(),
            "fixed_preload_targets_rad":fixed_preload.tolist() if fixed_preload is not None else None,
            "fixed_preload_native_caps_nm":fixed_caps.tolist() if fixed_caps is not None else None,
            "wrist_force_reference_n":wrist_force_reference,
            "contact_wrench_filter_time_constant_s":wrench_filter_tau,
            "filter_origin":"FIXED_DECLARED_FIXTURE; initialized empty, never reset during grasp/loading",
            "maximum_control_axis_torque_nm":maximum_control_torque,
            "bounded_torque_hold_completed":hold_completed,
            "axial_stiffness_probe":axial_probe,
            "measured_fixed_grip_settling":settling_result,
            "normal_15_n_planning_reference_exceeded":(max_pad>15.).tolist(),
            "original_hand_drive_cap_nm":1.,"original_arm_drive_cap_nm":100.,
            "explicit_closing_effort_command_cap_nm":command_cap if load_probe else None,
            "closing_position_drive_disabled_after_original_preload":actuation_target is not None and load_interface=="native_effort",
            "closing_native_pd_drive_cap_after_handover_nm":command_cap if actuation_target is not None and load_interface=="native_pd_drive" else 1.,
            "closing_native_pd_drive_caps_by_finger_nm":motor_caps.tolist() if actuation_target is not None and load_interface=="native_pd_drive" else [1.,1.,1.],
            "normal_feedback_configuration":normal_feedback,
            "normal_feedback_physical_tracking_or_capacity_verified":False,
            "actual_projected_reaction_stop_during_effort_load_nm":reaction_stop if load_probe else .9,
            "old_0p9_projected_reaction_reference_crossing_count":old_reaction_crossings,
            "source_cad_sized_force_load_probe":load_probe,"images":image_records,
            "physics_to_fabric_output_update_each_step":fabric_output_each_step,
            "sdk_standard_render_steps":bool(recipe.get("standard_render_steps",False)),
            "maximum_output_update_native_state_delta":maximum_output_sync_native_delta,
            "fabric_settings_at_start":fabric_settings,
            "articulation_kinematics_refreshed_before_render":refresh_articulation_before_render,
            "render_refresh_changed_native_state":render_refresh_state_violation,
            "free_space_bias_world_n_nm":bias.tolist(),"free_space_finger_effort_tare_nm":tare.tolist(),
            "loaded_rezeroing":False,"object_contact_truth_in_controller":False,
            "source_body_or_nut_pose_written_after_start":False,"hardware_authorized":False,
            "full_assembly_or_hardware_capacity_verified":False}
        (output/"mounted_grasp_result.json").write_text(json.dumps(result,indent=2)+"\n")
        np.savez_compressed(output/"mounted_grasp_summary.npz",values=np.asarray(samples))
    return result
