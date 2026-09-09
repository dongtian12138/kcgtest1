#!/usr/bin/env python3
"""Explicit laboratory slider/contact force-sensor balance, not assembly.

A finite PD slider presses a tip against a fixed plane. A fixed joint between
slider and tip is the six-axis sensor. Gravity and native contact impulses give
an independent wrench balance about that joint. No pose writes after start.
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--velocity-iterations", type=int, choices=(0, 1, 4), required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--mimic", action="store_true", help="Put a source-schema mimic constraint below the fixed wrist sensor")
parser.add_argument("--known-load-calibration", action="store_true", help="CPU six-axis calibration against declared weights and lever arms")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
if args.known_load_calibration:
    from te_known_wrench_calibration import run_known_load_calibration
    raise SystemExit(run_known_load_calibration(args.output, velocity_iterations=args.velocity_iterations))

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "multi_gpu": False})
failed = False
try:
    import carb
    import numpy as np
    import omni.usd
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from isaacsim.core.experimental.prims import RigidPrim
    from isaacsim.core.simulation_manager import SimulationManager
    from omni.physx.bindings._physx import SETTING_DISABLE_CONTACT_PROCESSING

    dt, tip_mass = 1/240., .1
    SimulationManager.set_physics_sim_device("cuda:0")
    carb.settings.get_settings().set_bool(SETTING_DISABLE_CONTACT_PROCESSING, False)
    world = World(stage_units_in_meters=1., physics_dt=dt, rendering_dt=1/60,
                  backend="numpy", device="cuda:0", sim_params={"use_gpu_pipeline": True})
    stage = omni.usd.get_context().get_stage()
    scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(world.get_physics_context().prim_path))
    scene.CreateSolverTypeAttr("TGS")
    scene.CreateMinVelocityIterationCountAttr(args.velocity_iterations)
    scene.CreateMaxVelocityIterationCountAttr(args.velocity_iterations)
    scene.CreateEnableExternalForcesEveryIterationAttr(True)
    UsdGeom.Xform.Define(stage, "/World/Lab")
    material = UsdShade.Material.Define(stage, "/World/LabMaterial")
    mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    mat.CreateStaticFrictionAttr(0.); mat.CreateDynamicFrictionAttr(0.); mat.CreateRestitutionAttr(0.)

    def rigid(path, xyz, mass):
        body = UsdGeom.Xform.Define(stage, path)
        body.AddTranslateOp().Set(Gf.Vec3d(*xyz))
        p = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(p)
        m = UsdPhysics.MassAPI.Apply(p)
        m.CreateMassAttr(mass); m.CreateDiagonalInertiaAttr(Gf.Vec3f(mass*.01**2/6))
        api = PhysxSchema.PhysxRigidBodyAPI.Apply(p)
        api.CreateLinearDampingAttr(0.); api.CreateAngularDampingAttr(0.); api.CreateSleepThresholdAttr(0.)
        PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0.)
        return p

    def collider(prim):
        UsdPhysics.CollisionAPI.Apply(prim)
        c = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        c.CreateContactOffsetAttr(.00005); c.CreateRestOffsetAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")

    rigid("/World/Lab/Base", [0.,0.,.05], 1.)
    if args.mimic:
        rigid("/World/Lab/Palm", [0.,0.,.04], .1)
    rigid("/World/Lab/Slider", [0.,0.,.02], .1)
    rigid("/World/Lab/Tip", [.02,0.,.006], tip_mass)
    sphere = UsdGeom.Sphere.Define(stage, "/World/Lab/Tip/ContactSphere")
    sphere.CreateRadiusAttr(.005); collider(sphere.GetPrim())
    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.CreateSizeAttr(1.); ground.AddTranslateOp().Set(Gf.Vec3d(.02,0.,-.005))
    ground.AddScaleOp().Set(Gf.Vec3f(.04,.04,.01)); collider(ground.GetPrim())
    clamp = UsdPhysics.FixedJoint.Define(stage, "/World/Lab/DeclaredBaseClamp")
    clamp.CreateBody1Rel().SetTargets(["/World/Lab/Base"])
    clamp.CreateLocalPos0Attr(Gf.Vec3f(0.,0.,.05))
    UsdPhysics.ArticulationRootAPI.Apply(clamp.GetPrim())
    art = PhysxSchema.PhysxArticulationAPI.Apply(clamp.GetPrim())
    art.CreateSolverPositionIterationCountAttr(32)
    art.CreateSolverVelocityIterationCountAttr(args.velocity_iterations)
    art.CreateSleepThresholdAttr(0.)
    slide = UsdPhysics.PrismaticJoint.Define(stage, "/World/Lab/SliderDrive")
    slide.CreateBody0Rel().SetTargets(["/World/Lab/Palm" if args.mimic else "/World/Lab/Base"])
    slide.CreateBody1Rel().SetTargets(["/World/Lab/Slider"])
    slide.CreateAxisAttr("Z"); slide.CreateLocalPos0Attr(Gf.Vec3f(0.,0.,-.02 if args.mimic else -.03))
    slide.CreateLowerLimitAttr(-.005); slide.CreateUpperLimitAttr(.005)
    drive = UsdPhysics.DriveAPI.Apply(slide.GetPrim(), "linear")
    drive.CreateTypeAttr("force"); drive.CreateStiffnessAttr(1000.); drive.CreateDampingAttr(10.)
    drive.CreateMaxForceAttr(5.); drive.CreateTargetPositionAttr(-.002); drive.CreateTargetVelocityAttr(0.)
    sensor = UsdPhysics.FixedJoint.Define(stage, "/World/Lab/WristSensor")
    sensor.CreateBody0Rel().SetTargets(["/World/Lab/Base" if args.mimic else "/World/Lab/Slider"])
    sensor.CreateBody1Rel().SetTargets(["/World/Lab/Palm" if args.mimic else "/World/Lab/Tip"])
    if args.mimic:
        sensor.CreateLocalPos0Attr(Gf.Vec3f(0.,0.,-.01))
        follower = UsdPhysics.PrismaticJoint.Define(stage, "/World/Lab/MimicFollower")
        follower.CreateBody0Rel().SetTargets(["/World/Lab/Slider"])
        follower.CreateBody1Rel().SetTargets(["/World/Lab/Tip"])
        follower.CreateAxisAttr("Z"); follower.CreateLocalPos0Attr(Gf.Vec3f(.02,0.,-.014))
        follower.GetPrim().AddAppliedSchema("NewtonMimicAPI")
        source_path = Path(__file__).resolve().parents[3] / "artifacts/kcg_connector/isaac/te_nail_tip_body_grasp_v1/handarm_original_nails_source_decomposition.usda"
        source_stage = Usd.Stage.Open(str(source_path))
        source_joint = source_stage.GetPrimAtPath("/handarm/Physics/f1j3")
        for attribute in source_joint.GetAttributes():
            if attribute.GetName().startswith("newton:mimic"):
                follower.GetPrim().CreateAttribute(attribute.GetName(),attribute.GetTypeName()).Set(attribute.Get())
        follower.GetPrim().CreateRelationship("newton:mimicJoint").SetTargets([slide.GetPath()])
    else:
        sensor.CreateLocalPos1Attr(Gf.Vec3f(-.02,0.,.014))
    tree = world.scene.add(SingleArticulation("/World/Lab", name="lab", reset_xform_properties=False))
    slider = world.scene.add(SingleRigidPrim("/World/Lab/Slider", name="slider", reset_xform_properties=False))
    tip = world.scene.add(SingleRigidPrim("/World/Lab/Tip", name="tip", reset_xform_properties=False))
    palm = (world.scene.add(SingleRigidPrim("/World/Lab/Palm",name="palm",reset_xform_properties=False))
            if args.mimic else None)
    contacts = RigidPrim(["/World/Lab/Tip"], resolve_paths=False,
                         contact_filter_paths=["/World/Ground"], max_contact_count=128)
    stage.GetRootLayer().Export(str(args.output / "before_reset.usda"))
    world.reset()
    row_index = tree._articulation_view._metadata.joint_indices["WristSensor"]+1

    def host(value):
        if hasattr(value, "detach"): return value.detach().cpu().numpy()
        return value.numpy() if hasattr(value, "numpy") else np.asarray(value)

    rows, joint_rows = [], []
    for step in range(480):
        world.step(render=False)
        origin = host((palm if args.mimic else slider).get_world_pose()[0]); center = host(tip.get_world_pose()[0])
        raw = host(tree.get_measured_joint_forces())[row_index]
        f, points, normals, _, counts, starts, _ = contacts.get_raw_contact_data()
        f, points, normals = f.numpy().ravel(), points.numpy(), normals.numpy()
        start, count = int(starts.numpy().ravel()[0]), int(counts.numpy().ravel()[0])
        sl = slice(start,start+count); forces=f[sl,None]*normals[sl]/dt
        expected = np.r_[forces.sum(0),np.cross(points[sl]-origin,forces).sum(0)]
        weight = np.array([0.,0.,-tip_mass*9.81])
        expected += np.r_[weight,np.cross(center-origin,weight)]
        if args.mimic:
            for body in (palm, slider):
                position=host(body.get_world_pose()[0]);gravity=np.array([0.,0.,-.1*9.81])
                expected += np.r_[gravity,np.cross(position-origin,gravity)]
        joint_rows.append(host(tree.get_joint_positions()).copy())
        rows.append([*raw,*expected,*(-raw-expected),*center,*host(tip.get_linear_velocity())])
    values=np.array(rows)
    result={"scope":"SYNTHETIC_CLOSED_CONTACT_JOINT_WRENCH_BALANCE_NOT_ROBOT_OR_ASSEMBLY",
            "velocity_iterations":args.velocity_iterations,"position_iterations":32,
            "external_forces_every_iteration":True,"dt_s":dt,"mimic_or_tendon_present":args.mimic,
            "finite_slider_drive_cap_n":5.,"contact_friction":0.,"post_start_pose_or_target_writes":False,
            "sensor_row":row_index,"sensor_frame_origin":"PALM_ORIGIN" if args.mimic else "SLIDER_ORIGIN_FIXED_JOINT_CHILD_ANCHOR",
            "joint_names":list(tree._articulation_view._metadata.joint_names),
            "last_joint_positions":np.asarray(joint_rows[-1]).tolist(),
            "columns":"raw_incoming[6], expected_gravity_plus_contact[6], balance_error[6], tip_position[3], tip_velocity[3]",
            "last_half_second_mean":values[-120:].mean(0).tolist(),
            "last_half_second_tip_position_range_m":np.ptp(values[-120:,18:21],axis=0).tolist()}
    np.savez_compressed(args.output/"samples.npz",values=values,joint_positions=np.asarray(joint_rows))
    (args.output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)
except Exception:
    failed=True
    import traceback
    error=traceback.format_exc();(args.output/"error.txt").write_text(error);print(error,flush=True)
finally:
    app.close(exit_code=1 if failed else 0)
