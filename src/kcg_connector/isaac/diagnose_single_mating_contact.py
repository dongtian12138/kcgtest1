#!/usr/bin/env python3
"""Drive the unchanged source pin through one explicit spring-contact coupon.

A declared prismatic laboratory guide moves the source Body via a finite native
drive. This is an isolated force-law experiment, not connector assembly evidence.
"""
import argparse
import json
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--stiffness-n-m",type=float,default=1500.)
parser.add_argument("--assembled-thread-scene",type=Path,
                    help="Use an already authored connector scene in an explicit rotary/axial laboratory guide; no robot or assembly claim.")
parser.add_argument("--thread-capture-check",action="store_true",
                    help="Only a short60-degree/3-second pre-contact laboratory stroke to check image/physics synchronization.")
parser.add_argument("--insert-duration-s",type=float,choices=(2.,35.),default=2.,
                    help="One bounded rate comparison, retaining the original force-law coupon and drive.")
parser.add_argument("--insert-only",action="store_true",
                    help="End after inserted hold when only insertion-rate response is being diagnosed; original withdrawal results remain separate.")
parser.add_argument("--all-source-contacts",action="store_true",
                    help="Replicate the same validated contact envelope at all 128 source cavity centres; still a guided laboratory coupon.")
args=parser.parse_args()
if args.thread_capture_check and not args.assembled_thread_scene:
    parser.error("--thread-capture-check requires the explicit assembled-thread laboratory scene")
args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
from isaacsim import SimulationApp
app=SimulationApp({"headless":True,"multi_gpu":False,"fast_shutdown":True})
if args.assembled_thread_scene:
    failed=False
    try:
        from te_assembled_thread_load_probe import run_assembled_thread_load_probe
        run_assembled_thread_load_probe(args.assembled_thread_scene.resolve(),args.output,
                                       capture_check=args.thread_capture_check)
    except Exception:
        import traceback
        failed=True;error=traceback.format_exc()
        (args.output/"error.txt").write_text(error);print(error,flush=True)
    finally:
        app.close(exit_code=1 if failed else 0)
    raise SystemExit(1 if failed else 0)
failed=False
try:
    import carb
    import numpy as np
    import omni.usd
    from pxr import Gf,PhysxSchema,Usd,UsdGeom,UsdPhysics,UsdShade,Vt
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleRigidPrim
    from isaacsim.core.experimental.prims import RigidPrim
    from isaacsim.core.simulation_manager import SimulationManager
    from omni.physx.bindings._physx import SETTING_DISABLE_CONTACT_PROCESSING
    from te_mating_contact_scene import representative_clip_mesh

    repo=Path(__file__).resolve().parents[3];dt=1/240.
    if not np.isfinite(args.stiffness_n_m) or args.stiffness_n_m<=0:
        raise ValueError("positive finite diagnostic contact stiffness required")
    source=repo/"artifacts/kcg_connector/isaac/te_full_assembly_20260905/grounding_band_geometry_03/rigid_body.npz"
    data=np.load(source)
    mass_path=repo/"artifacts/kcg_connector/isaac/te_j35_free_split_tabletop_real_mass_resistance_0p020_v2/TE_J35_FREE_SPLIT_PLUG_V1.usdc"
    mass_stage=Usd.Stage.Open(str(mass_path));original=mass_stage.GetPrimAtPath("/TE_J35FreeSplitPlug/Body")
    masses={name:original.GetAttribute(name).Get() for name in
            ["physics:mass","physics:centerOfMass","physics:diagonalInertia","physics:principalAxes"]}
    mass=float(masses["physics:mass"])
    SimulationManager.set_physics_sim_device("cuda:0")
    carb.settings.get_settings().set_bool(SETTING_DISABLE_CONTACT_PROCESSING,False)
    world=World(stage_units_in_meters=1.,physics_dt=dt,rendering_dt=1/60,
                backend="numpy",device="cuda:0",sim_params={"use_gpu_pipeline":True})
    stage=omni.usd.get_context().get_stage()
    physics=UsdPhysics.Scene.Get(stage,world.get_physics_context().prim_path)
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0.,0.,-1.));physics.CreateGravityMagnitudeAttr(9.81)
    scene=PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(world.get_physics_context().prim_path))
    scene.CreateSolverTypeAttr("TGS");scene.CreateEnableExternalForcesEveryIterationAttr(True)
    scene.CreateMinVelocityIterationCountAttr(1);scene.CreateMaxVelocityIterationCountAttr(1)
    def material(path,stiffness):
        m=UsdShade.Material.Define(stage,path);a=UsdPhysics.MaterialAPI.Apply(m.GetPrim())
        a.CreateStaticFrictionAttr(.45);a.CreateDynamicFrictionAttr(.45);a.CreateRestitutionAttr(0.)
        px=PhysxSchema.PhysxMaterialAPI.Apply(m.GetPrim());px.CreateFrictionCombineModeAttr("min")
        if stiffness:
            px.CreateCompliantContactStiffnessAttr(stiffness)
            px.CreateCompliantContactDampingAttr(0.)
            px.CreateCompliantContactAccelerationSpringAttr(False)
        return m
    hard=material("/World/OriginalFriction",0.);soft=material("/World/RepresentativeClipMaterial",args.stiffness_n_m)
    def mesh(path,vertices,faces,mat,sdf=False):
        m=UsdGeom.Mesh.Define(stage,path)
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(vertices,np.float32)))
        m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces),3,np.int32)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(faces,np.int32).ravel()))
        m.CreateSubdivisionSchemeAttr("none")
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr("sdf" if sdf else "none")
        c=PhysxSchema.PhysxCollisionAPI.Apply(m.GetPrim());c.CreateContactOffsetAttr(.00005);c.CreateRestOffsetAttr(0.)
        PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(m.GetPrim()).CreateWeldToleranceAttr(0.)
        if sdf:
            a=PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(m.GetPrim());a.CreateSdfResolutionAttr(1024)
            a.CreateSdfSubgridResolutionAttr(6);a.CreateSdfNarrowBandThicknessAttr(.002)
            a.CreateSdfTriangleCountReductionFactorAttr(1.)
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat,materialPurpose="physics")
        return m
    start_depth=.0103;end_depth=.0140
    lab=UsdGeom.Xform.Define(stage,"/World/LaboratoryAssembly")
    body_path="/World/LaboratoryAssembly/SourceBody"
    body=UsdGeom.Xform.Define(stage,body_path)
    body.AddTranslateOp().Set(Gf.Vec3d(0.,0.,-start_depth))
    body.AddOrientOp().Set(Gf.Quatf(0.,Gf.Vec3f(0.,1.,0.)))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim());UsdPhysics.MassAPI.Apply(body.GetPrim())
    for name,value in masses.items():body.GetPrim().CreateAttribute(name,original.GetAttribute(name).GetTypeName()).Set(value)
    mesh(body_path+"/OriginalRigidCore",data["vertices_m"],data["faces"],hard,True)
    phys=PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim());phys.CreateSleepThresholdAttr(0.)
    phys.CreateLinearDampingAttr(0.);phys.CreateAngularDampingAttr(0.)
    phys.CreateSolverPositionIterationCountAttr(32);phys.CreateSolverVelocityIterationCountAttr(1)
    PhysxSchema.PhysxContactReportAPI.Apply(body.GetPrim()).CreateThresholdAttr(0.)
    vertices,faces,geometry=representative_clip_mesh()
    centers=json.loads((repo/"artifacts/kcg_connector/isaac/te_full_assembly_20260905/source_socket_blind_bore_faces_v1.json").read_text())["cap_faces"]
    selected_centers=(np.asarray([r["center_m"][:2] for r in centers]) if args.all_source_contacts else
                      np.asarray([min(centers,key=lambda r:np.linalg.norm(r["center_m"][:2]))["center_m"][:2]]))
    contact_count=len(selected_centers)
    if contact_count not in (1,128):
        raise RuntimeError("the coupon must represent one contact or all 128 source contacts")
    clip_paths=[]
    for i,center in enumerate(selected_centers):
        translated=vertices.copy();translated[:,:2]+=center
        path=(f"/World/DeclaredContactClip{i:03d}" if args.all_source_contacts else "/World/DeclaredSingleContactClip")
        mesh(path,translated,faces,soft);clip_paths.append(path)
    np.savez_compressed(args.output/"representative_clip.npz",local_vertices_m=vertices,faces=faces,
                        source_contact_centers_socket_xy_m=selected_centers)
    joint=UsdPhysics.PrismaticJoint.Define(stage,"/World/LaboratoryAssembly/DeclaredAxialLaboratoryGuide")
    joint.CreateBody1Rel().SetTargets([body.GetPath()]);joint.CreateAxisAttr("Z")
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.,0.,-start_depth))
    joint.CreateLocalRot1Attr(Gf.Quatf(0.,Gf.Vec3f(0.,-1.,0.)))
    joint.CreateLowerLimitAttr(-.004);joint.CreateUpperLimitAttr(.0005)
    drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),"linear")
    # Scale only the explicit test apparatus to keep comparable tracking error
    # as contacts are replicated; no hand/arm limit is changed or tested here.
    stiffness=100000.*contact_count;drive_cap=5.*contact_count
    drive.CreateTypeAttr("force");drive.CreateStiffnessAttr(stiffness)
    drive.CreateDampingAttr(100.*np.sqrt(contact_count));drive.CreateMaxForceAttr(drive_cap)
    gravity_bias=mass*9.81/stiffness
    drive.CreateTargetPositionAttr(gravity_bias);drive.CreateTargetVelocityAttr(0.)
    PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim()).CreateJointFrictionAttr(0.)
    contact=RigidPrim([body_path],resolve_paths=False,
                      contact_filter_paths=clip_paths,max_contact_count=16384 if contact_count>1 else 4096)
    part=SingleRigidPrim(body_path,name="source_body",reset_xform_properties=False)
    stage.GetRootLayer().Export(str(args.output/"coupon_before_physics.usda"))
    world.reset();part.initialize()
    def host(v):return v.detach().cpu().numpy() if hasattr(v,"detach") else v.numpy() if hasattr(v,"numpy") else np.asarray(v)
    def blend(u):return 10*u**3-15*u**4+6*u**5
    rows=[];phases=[];pose_rows=[]
    sequence=[("initial_hold",.25,0.,0.),("insert",args.insert_duration_s,0.,end_depth-start_depth),
              ("inserted_hold",.4,end_depth-start_depth,end_depth-start_depth)]
    if not args.insert_only:
        sequence.extend([("withdraw",args.insert_duration_s,end_depth-start_depth,0.),("final_hold",.25,0.,0.)])
    for phase,duration,x0,x1 in sequence:
        count=round(duration/dt)
        for i in range(count):
            command=x0+(x1-x0)*blend((i+1)/count)
            drive.GetTargetPositionAttr().Set(float(-command+gravity_bias))
            world.step(render=False)
            p,quat=part.get_world_pose();p=host(p);quat=host(quat);actual_depth=-float(p[2])
            pose_rows.append(np.r_[p,quat])
            f,points,normals,separations,counts,starts,_=contact.get_raw_contact_data()
            start,n=int(starts.numpy()[0]),int(counts.numpy()[0]);sl=slice(start,start+n)
            F=f.numpy()[sl].reshape(-1,1)*normals.numpy()[sl]/dt
            friction,_,c,st=contact.get_friction_data();friction=friction.numpy();Ft=np.zeros(3)
            for friction_count,friction_start in zip(c.numpy().ravel(),st.numpy().ravel()):
                friction_start,friction_count=int(friction_start),int(friction_count)
                Ft+=friction[friction_start:friction_start+friction_count].sum(0)/dt
            q=float(p[2]+start_depth)
            rows.append([len(rows)*dt,start_depth+command,actual_depth,float(q),n,*F.sum(0),*Ft,
                         float(np.linalg.norm(F,axis=1).sum()),float(np.max(np.maximum(-separations.numpy()[sl],0))) if n else 0.])
            phases.append(phase)
    values=np.asarray(rows);axial=values[:,7]+values[:,10]
    result={"scope":"SINGLE_CONTACT_SOURCE_PIN_FORCE_COUPON_NOT_ASSEMBLY","geometry":geometry,
            "source_body_geometry":str(source),"source_body_mass_kg":mass,"source_body_mass_inertia_geometry_preserved":True,
            "contact_count":contact_count,"clip_centers_socket_xy_m":selected_centers.tolist(),
            "physics_dt_s":dt,"solver":"GPU_TGS_32_1_EXTERNAL_EVERY_ITERATION",
            "compliant_per_contact_stiffness_n_m":args.stiffness_n_m,"compliant_damping_ns_m":0.,"contact_friction":.45,
            "finite_axial_drive_cap_n":drive_cap,"guide_is_an_explicit_laboratory_fixture":True,
            "laboratory_drive_scaled_with_contact_count_not_a_robot_force_increase":True,
            "guide_representation":"MAXIMAL_COORDINATE_PRISMATIC_NATIVE_DRIVE",
            "post_start_body_pose_writes":False,"hardware_authorized":False,
            "insert_duration_s":args.insert_duration_s,"withdrawal_executed":not args.insert_only,
            "reference_mean_engagement_n":1.,"reference_mean_separation_n":.4,
            "reference_is_other_manufacturer_not_exact_TE_calibration":True,
            "maximum_insert_contact_axial_force_n":float(axial[np.array(phases)=="insert"].max()),
            "maximum_withdraw_contact_axial_force_magnitude_n":(float(-axial[np.array(phases)=="withdraw"].min())
                                                                if not args.insert_only else None),
            "maximum_native_contact_count":int(values[:,4].max()),
            "maximum_drive_tracking_error_m":float(np.max(abs(values[:,1]-values[:,2]))),
            "maximum_native_lateral_drift_m":float(np.max(np.linalg.norm(np.array(pose_rows)[:,:2],axis=1))),
            "columns":["time_s","commanded_body_depth_m","actual_body_depth_m","guide_displacement_from_body_pose_m","normal_contact_count",
                       "normal_fx_n","normal_fy_n","normal_fz_n","friction_fx_n","friction_fy_n","friction_fz_n",
                       "normal_magnitude_sum_n","maximum_compliant_penetration_m"]}
    np.savez_compressed(args.output/"samples.npz",values=values,phases=np.asarray(phases),body_poses=np.asarray(pose_rows))
    (args.output/"result.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2),flush=True)
except Exception:
    failed=True
    import traceback
    error=traceback.format_exc();(args.output/"error.txt").write_text(error);print(error,flush=True)
finally:
    app.close(exit_code=1 if failed else 0)
