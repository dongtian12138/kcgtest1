"""Measure a saved connector model using a declared native laboratory actuator.

The external joint guides only lateral position and tilt of the Nut. Translation
along its axis is free; the original contact geometry must produce all axial
advance. This guide and actuator are confined to this diagnostic, not the robot.
"""
import json
from pathlib import Path
from time import perf_counter

import numpy as np


def run_assembled_thread_load_probe(source_scene,output,*,capture_check=False):
    import carb
    import omni.usd
    import omni.replicator.core as rep
    from pxr import Gf,PhysxSchema,Sdf,Usd,UsdGeom,UsdLux,UsdPhysics
    from scipy.spatial.transform import Rotation
    from isaacsim.core.api import World
    from isaacsim.core.experimental.prims import RigidPrim
    from isaacsim.core.simulation_manager import SimulationManager
    from omni.physx.bindings._physx import SETTING_DISABLE_CONTACT_PROCESSING
    from te_foundationpose_handoff_runtime import _author_camera,_camera_cv_pose_from_eye_target
    import cv2

    output=Path(output);started=perf_counter();dt=1/240.
    report=json.loads(source_scene.with_name("assembly_scene.json").read_text())
    if report["representative_mating_contacts"]["contact_count"]!=128:
        raise ValueError("the saved source must contain all 128 finite mating contacts")
    source=Usd.Stage.Open(str(source_scene));source_layer=source.GetRootLayer()
    SimulationManager.set_physics_sim_device("cuda:0")
    carb.settings.get_settings().set_bool(SETTING_DISABLE_CONTACT_PROCESSING,False)
    world=World(stage_units_in_meters=1.,physics_dt=dt,rendering_dt=1/60,
                backend="numpy",device="cuda:0",sim_params={"use_gpu_pipeline":True})
    stage=omni.usd.get_context().get_stage();layer=stage.GetRootLayer()
    UsdGeom.Xform.Define(stage,"/World")
    # Copy the already validated connector authoring, preserving bindings,
    # filters, original masses and inertia. The lab contains no hand or arm.
    copied=[]
    for prim in source.GetPrimAtPath("/World").GetChildren():
        if prim.GetName() in ("HandArm","FixtureMaterial"):continue
        path=prim.GetPath()
        if not Sdf.CopySpec(source_layer,path,layer,path):raise RuntimeError(f"copy failed: {path}")
        copied.append(str(path))
    physics=PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(world.get_physics_context().prim_path))
    physics.CreateSolverTypeAttr("TGS");physics.CreateEnableExternalForcesEveryIterationAttr(True)
    physics.CreateMinVelocityIterationCountAttr(1);physics.CreateMaxVelocityIterationCountAttr(1)
    body_path="/World/TE_J35FreeSplitPlug/Body";nut_path="/World/TE_J35FreeSplitPlug/CouplingNut"
    body=stage.GetPrimAtPath(body_path);nut=stage.GetPrimAtPath(nut_path)
    source_mass={str(p.GetPath()):{n:str(p.GetAttribute(n).Get()) for n in
                 ("physics:mass","physics:centerOfMass","physics:diagonalInertia","physics:principalAxes")}
                 for p in (body,nut)}
    def transform(prim):return np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    initial_nut=transform(nut);origin=initial_nut[:3,3];frame=initial_nut[:3,:3];axis=frame[:,2]
    q0=Rotation.from_matrix(frame).as_quat();socket=np.asarray(report["socket_initial_position_world_m"])
    guide=UsdPhysics.Joint.Define(stage,"/World/DeclaredLaboratoryRotaryAxialGuide")
    guide.CreateBody1Rel().SetTargets([nut.GetPath()]);guide.CreateExcludeFromArticulationAttr(True)
    guide.CreateLocalPos0Attr(Gf.Vec3f(*origin));guide.CreateLocalRot0Attr(Gf.Quatf(q0[3],Gf.Vec3f(*q0[:3])))
    guide.CreateLocalPos1Attr(Gf.Vec3f(0.));guide.CreateLocalRot1Attr(Gf.Quatf(1.))
    for name in ("transX","transY","rotX","rotY"):
        limit=UsdPhysics.LimitAPI.Apply(guide.GetPrim(),name)
        limit.CreateLowAttr(1.);limit.CreateHighAttr(-1.)
    # No axial drive and no kinematic lead/displacement constraint.
    drive=UsdPhysics.DriveAPI.Apply(guide.GetPrim(),"rotZ")
    K=30.;D=.1;cap=1.6
    drive.CreateTypeAttr("force");drive.CreateStiffnessAttr(K*np.pi/180.)
    drive.CreateDampingAttr(D*np.pi/180.);drive.CreateMaxForceAttr(cap)
    drive.CreateTargetPositionAttr(0.);drive.CreateTargetVelocityAttr(0.)
    for p in (body,nut):
        PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0.)
    filters=[report["collision"]["receptacle_collision"],*report["representative_mating_contacts"]["clip_paths"]]
    contacts=RigidPrim([body_path,nut_path],resolve_paths=False,contact_filter_paths=filters,max_contact_count=16384)
    camera_path="/World/LaboratoryEvidenceCamera"
    _author_camera(stage,camera_path,_camera_cv_pose_from_eye_target(socket+[.12,-.14,.11],socket+[0,0,.01]),
        resolution=(800,600),focal_length_mm=32.,horizontal_aperture_mm=36.,clipping_range_m=(.01,5.),Gf=Gf,UsdGeom=UsdGeom)
    UsdLux.DomeLight.Define(stage,"/World/LaboratoryEvidenceLight").CreateIntensityAttr(1000.)
    product=rep.create.render_product(camera_path,(800,600));rgb=rep.AnnotatorRegistry.get_annotator("rgb");rgb.attach([product.path])
    layer.Export(str(output/"laboratory_scene_before_physics.usda"))
    world.reset()
    def host(a):return a.detach().cpu().numpy() if hasattr(a,"detach") else a.numpy() if hasattr(a,"numpy") else np.asarray(a)
    def poses():return np.concatenate([host(x).ravel() for x in contacts.get_world_poses()])
    image_rows=[]
    def capture(name):
        import omni.timeline
        from omni.physxfabric import get_physx_fabric_interface
        before=poses().copy();time_before=float(world.current_time);was_playing=bool(world.is_playing())
        if SimulationManager.is_fabric_enabled():get_physx_fabric_interface().force_update(dt,time_before)
        timeline=omni.timeline.get_timeline_interface();auto=timeline.is_auto_updating()
        settings=carb.settings.get_settings();play=settings.get("/app/player/playSimulations")
        try:
            timeline.set_auto_update(False);timeline.commit_silently();settings.set("/app/player/playSimulations",False)
            for _ in range(3):omni.kit.app.get_app().update()
            # Match the existing assembly recorder: normal graph evaluation
            # is restored for the explicit zero-time render. Do not pause a
            # moving episode just to render, or leave the player disabled.
            settings.set("/app/player/playSimulations",play)
            rep.orchestrator.step(rt_subframes=1,delta_time=0.,pause_timeline=not was_playing)
        finally:
            settings.set("/app/player/playSimulations",play);timeline.set_auto_update(auto);timeline.commit_silently()
        rgba=np.asarray(rgb.get_data());delta=float(np.max(abs(poses()-before)))
        if rgba.ndim!=3 or not rgba.size or delta>0 or world.current_time!=time_before:
            raise RuntimeError("laboratory evidence capture missing or changed native pose/time")
        cv2.imwrite(str(output/(name+".png")),cv2.cvtColor(rgba[:,:,:3],cv2.COLOR_RGB2BGR))
        image_rows.append({"image":name+".png","time_s":time_before,"maximum_native_pose_delta":delta})
        if was_playing and not world.is_playing():world.play()
    capture("initial_lab_state")
    records=[];peak_raw=None;peak_load=-1.;previous=0.;unwrapped=0.
    stream=(output/"samples.jsonl").open("x",buffering=1)
    # A single finite stroke traverses the contact band without approaching the
    # unimplemented seal constitutive region. Commands are time-based only.
    # Positive rotation in the initial Nut frame advances the source right-hand
    # helix along its local +Z (approximately world -Z). The lab frame is not
    # the world-Z convention used by the assembly's reported yaw.
    duration,target=((3.,60.) if capture_check else (35.,175.));hold=.25;end_hold=.5
    for step in range(round((hold+duration+end_hold)/dt)):
        t=(step+1)*dt;u=np.clip((t-hold)/duration,0.,1.)
        command=target*(10*u**3-15*u**4+6*u**5)
        drive.GetTargetPositionAttr().Set(float(command));world.step(render=False)
        positions,quats=(host(x) for x in contacts.get_world_poses())
        R=Rotation.from_quat(quats[1,[1,2,3,0]]).as_matrix();relative=frame.T@R
        wrapped=float(np.arctan2(relative[1,0],relative[0,0]));unwrapped+=float(np.arctan2(np.sin(wrapped-previous),np.cos(wrapped-previous)));previous=wrapped
        f,p,n,separation,counts,starts,ids=contacts.get_raw_contact_data()
        f,p,n,separation,counts,starts=(host(x) for x in (f,p,n,separation,counts,starts));f=f.ravel();separation=separation.ravel()
        normal_wrenches=[];point_counts=[];normal_load=[];depth=[]
        used=[]
        for index,(count,start) in enumerate(zip(counts.ravel(),starts.ravel())):
            count,start=int(count),int(start);sl=slice(start,start+count);F=f[sl,None]*n[sl]/dt
            normal_wrenches.append(np.r_[F.sum(0),np.cross(p[sl]-positions[index],F).sum(0)])
            point_counts.append(count);normal_load.append(float(np.linalg.norm(F,axis=1).sum()))
            depth.append(float(np.maximum(-separation[sl],0).max()) if count else 0.);used.extend(range(start,start+count))
        fr,fp,fc,fs=(host(x) for x in contacts.get_friction_data());fr=fr/dt
        friction_wrenches=np.zeros((2,6));fr_used=[]
        # Friction data is one header per sensor/filter pair.
        for index in range(2):
            for count,start in zip(fc[index].ravel(),fs[index].ravel()):
                count,start=int(count),int(start);sl=slice(start,start+count)
                friction_wrenches[index]+=np.r_[fr[sl].sum(0),np.cross(fp[sl]-positions[index],fr[sl]).sum(0)]
                fr_used.extend(range(start,start+count))
        w=np.asarray(normal_wrenches)+friction_wrenches
        body_depth=float(socket[2]-positions[0,2]);nut_axis=float(w[1,3:]@axis)
        spring_estimate=float(np.clip(K*(np.deg2rad(command)-unwrapped),-cap,cap))
        row={"step":step,"time_s":float(world.current_time),"command_deg":float(command),
             "record_only_nut_rotation_deg":float(np.rad2deg(unwrapped)),"record_only_body_depth_m":body_depth,
             "record_only_positions_world_m":positions.tolist(),"record_only_quaternions_wxyz":quats.tolist(),
             "record_only_normal_wrenches_n_nm":np.asarray(normal_wrenches).tolist(),
             "record_only_friction_wrenches_n_nm":friction_wrenches.tolist(),
             "record_only_normal_counts":point_counts,"record_only_normal_loads_n":normal_load,
             "record_only_maximum_penetration_m":depth,"record_only_nut_contact_axis_torque_nm":nut_axis,
             "posthoc_rotary_spring_effort_without_damping_nm":spring_estimate}
        stream.write(json.dumps(row,separators=(",",":"))+"\n");records.append(row)
        if abs(nut_axis)>peak_load:
            peak_load=abs(nut_axis);used=np.asarray(used,int);fr_used=np.asarray(fr_used,int)
            peak_raw={"step":step,"point_world_m":p[used].copy(),"normal_world":n[used].copy(),
                      "normal_impulse_ns":f[used].copy(),"separation_m":separation[used].copy(),
                      "normal_counts":counts.copy(),"normal_starts":starts.copy(),
                      "friction_force_world_n":fr[fr_used].copy(),"friction_point_world_m":fp[fr_used].copy(),
                      "friction_counts":fc.copy(),"friction_starts":fs.copy()}
    stream.close();capture("final_lab_state");world.pause()
    np.savez_compressed(output/"peak_torque_raw_contacts.npz",**peak_raw)
    result={"scope":"EXPLICIT_ROTARY_AXIAL_LAB_GUIDE_NOT_ROBOT_ASSEMBLY","source_scene":str(source_scene),
        "copied_source_roots":copied,"original_plug_mass_properties":source_mass,
        "original_internal_joint_and_contacts_preserved":True,"robot_present":False,
        "explicit_external_guide_locks":["Nut lateral position","Nut axis tilt"],"axial_drive":False,
        "lead_displacement_constraint":False,"post_start_object_pose_writes":False,
        "controller_inputs":"TIME_ONLY_NATIVE_FINITE_ROTARY_DRIVE","laboratory_torque_cap_nm":cap,
        "short_capture_check_only":capture_check,"command_duration_s":duration,"command_angle_deg":target,
        "laboratory_stiffness_nm_rad":K,"laboratory_damping_nm_s_rad":D,"hand_or_arm_caps_changed":False,
        "actual_final_nut_rotation_deg":records[-1]["record_only_nut_rotation_deg"],
        "initial_body_depth_m":records[0]["record_only_body_depth_m"],"final_body_depth_m":body_depth,
        "maximum_nut_socket_contact_axis_torque_nm":peak_load,
        "maximum_body_contact_axial_force_n":max(r["record_only_normal_wrenches_n_nm"][0][2]+r["record_only_friction_wrenches_n_nm"][0][2] for r in records),
        "physics_dt_s":dt,"wall_seconds":perf_counter()-started,"images":image_rows,
        "full_assembly_or_hardware_verified":False}
    (output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)
