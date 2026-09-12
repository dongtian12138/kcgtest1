"""Bounded CPU test of the measured finger's internal four-bar constraint.

This is a two-link mechanism coupon, not a grasp or assembly acceptance. The
68 mm connecting rod is represented by a distance constraint. The two existing
hinges retain their degrees of freedom; only the proximal hinge is driven.
"""
import argparse
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-limit-s", type=float, default=90.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[3] / "artifacts/hand_mechanism_audit_20260912/linkage_independent_check.json"
    measured = json.loads(source.read_text())
    p = {name: np.asarray(value) * .001 for name, value in measured["anchors"].items()}
    u = (p["B"] - p["O"]) / np.linalg.norm(p["B"] - p["O"])
    normal = np.array([1., -1., 0.]) / math.sqrt(2.)
    v = np.cross(normal, u)
    project = lambda point: np.array([(point-p["O"])@u, (point-p["O"])@v])
    a, b, c = (project(p[name]) for name in ("A", "B", "C"))
    length = float(np.linalg.norm(b))
    distal = float(np.linalg.norm(c-b))
    rod = float(np.linalg.norm(c-a))
    phi0 = float(np.arctan2(*(c-b)[::-1]))
    branch_sign = float(np.sign((a-b)[0]*(c-b)[1]-(a-b)[1]*(c-b)[0]))

    def expected(q):
        bn = length*np.array([math.cos(q), math.sin(q)])
        ac = a-bn
        d = float(np.linalg.norm(ac))
        e = ac/d
        x = (distal**2-rod**2+d*d)/(2*d)
        h2 = distal**2-x*x
        if h2 <= 0:
            raise ValueError("Requested probe motion leaves the measured assembly branch")
        mid = bn+x*e
        perpendicular = np.array([-e[1], e[0]])
        cn = mid+branch_sign*math.sqrt(h2)*perpendicular
        phi = math.atan2(cn[1]-bn[1], cn[0]-bn[0])
        return math.atan2(math.sin(phi-phi0-q), math.cos(phi-phi0-q))

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "hide_ui": True})
    try:
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, Sdf
        from isaacsim.core.api import World
        from isaacsim.core.experimental.prims import Articulation
        from isaacsim.core.simulation_manager import SimulationManager
        SimulationManager.set_physics_sim_device("cpu")
        dt = 1/960.
        world = World(stage_units_in_meters=1., physics_dt=dt, rendering_dt=1/60., backend="numpy", device="cpu")
        stage = omni.usd.get_context().get_stage()
        scene = UsdPhysics.Scene(stage.GetPrimAtPath(world.get_physics_context().prim_path))
        scene.CreateGravityMagnitudeAttr(0.)
        phys = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        phys.CreateEnableGPUDynamicsAttr(False)
        phys.CreateBroadphaseTypeAttr("MBP")
        phys.CreateSolverTypeAttr("TGS")
        phys.CreateMinPositionIterationCountAttr(64)
        phys.CreateMaxPositionIterationCountAttr(64)
        phys.CreateMinVelocityIterationCountAttr(4)
        phys.CreateMaxVelocityIterationCountAttr(4)
        root = UsdGeom.Xform.Define(stage, "/World/Finger")
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
        articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
        articulation_api.CreateSolverPositionIterationCountAttr(64)
        articulation_api.CreateSolverVelocityIterationCountAttr(4)

        def quat(angle):
            return Gf.Quatf(math.cos(angle/2), Gf.Vec3f(0, 0, math.sin(angle/2)))

        def body(name, pos, angle, mass, inertia, size, center, color):
            path = "/World/Finger/"+name
            xform = UsdGeom.Xform.Define(stage, path)
            xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
            xform.AddOrientOp().Set(quat(angle))
            UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
            m = UsdPhysics.MassAPI.Apply(xform.GetPrim())
            m.CreateMassAttr(mass)
            m.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
            m.CreateCenterOfMassAttr(Gf.Vec3f(*center))
            box = UsdGeom.Cube.Define(stage, path+"/display")
            box.CreateSizeAttr(1.)
            box.AddTranslateOp().Set(Gf.Vec3d(*center))
            box.AddScaleOp().Set(Gf.Vec3f(*size))
            box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            return path

        base = body("Base", (0,0,0), 0, 1., (.001,.001,.001), (.015,.015,.008), (0,0,0), (.5,.5,.5))
        near = body("Near", (0,0,0), 0, .073035, (2.3141e-5,2.3914e-5,4.4461e-5), (length,.006,.006), (length/2,0,0), (.1,.4,.8))
        tip = body("Distal", (*b,0), phi0, .057879, (1.6904e-5,1.2703e-5,2.4528e-5), (distal,.006,.006), (distal/2,0,0), (.9,.6,.1))
        fixed = UsdPhysics.FixedJoint.Define(stage, "/World/Finger/FixedBase")
        fixed.CreateBody1Rel().SetTargets([Sdf.Path(base)])
        for name, parent, child, anchor, offset in (("proximal",base,near,(0,0,0),0.),("distal",near,tip,(*b,0),phi0)):
            j = UsdPhysics.RevoluteJoint.Define(stage, "/World/Finger/"+name)
            j.CreateBody0Rel().SetTargets([Sdf.Path(parent)])
            j.CreateBody1Rel().SetTargets([Sdf.Path(child)])
            j.CreateAxisAttr("Z")
            j.CreateLocalPos0Attr(Gf.Vec3f(*anchor))
            j.CreateLocalPos1Attr(Gf.Vec3f(0))
            j.CreateLocalRot0Attr(quat(offset))
            j.CreateLocalRot1Attr(quat(0))
            j.CreateLowerLimitAttr(-90.)
            j.CreateUpperLimitAttr(90.)
            state = PhysxSchema.JointStateAPI.Apply(j.GetPrim(), "angular")
            state.CreatePositionAttr(0.)
            state.CreateVelocityAttr(0.)
            drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(math.radians(120.) if name=="proximal" else 0.)
            drive.CreateDampingAttr(math.radians(2.) if name=="proximal" else 0.)
            drive.CreateMaxForceAttr(3.5 if name=="proximal" else 0.)
        loop = UsdPhysics.DistanceJoint.Define(stage, "/World/Finger/MeasuredRod")
        loop.CreateBody0Rel().SetTargets([Sdf.Path(base)])
        loop.CreateBody1Rel().SetTargets([Sdf.Path(tip)])
        loop.CreateLocalPos0Attr(Gf.Vec3f(*a,0))
        loop.CreateLocalPos1Attr(Gf.Vec3f(distal,0,0))
        loop.CreateMinDistanceAttr(rod)
        loop.CreateMaxDistanceAttr(rod)
        loop.CreateExcludeFromArticulationAttr(True)
        loop.CreateCollisionEnabledAttr(False)
        stage.GetRootLayer().Export(str(args.output/"scene.usda"))
        start = time.monotonic()
        world.reset()
        robot = Articulation("/World/Finger")
        names = list(robot.dof_names)
        pi, di = names.index("proximal"), names.index("distal")
        robot.set_dof_gains(np.array([120. if name=="proximal" else 0. for name in names]),np.array([2. if name=="proximal" else 0. for name in names]),indices=0)
        robot.set_dof_max_efforts(np.array([3.5 if name=="proximal" else 0. for name in names]),indices=0)
        samples = []
        abort = None
        for i in range(3360):
            t = i*dt
            if time.monotonic()-start > args.wall_limit_s:
                abort = "WALL_LIMIT"; break
            if t < .25: goal=0.
            elif t < 1.25:
                f=t-.25;goal=.2*(10*f**3-15*f**4+6*f**5)
            elif t < 2.25:
                f=t-1.25;goal=.2-.35*(10*f**3-15*f**4+6*f**5)
            elif t < 3.25:
                f=t-2.25;goal=-.15+.15*(10*f**3-15*f**4+6*f**5)
            else: goal=0.
            target=np.zeros((1,2));target[0,pi]=goal
            robot.set_dof_position_targets(target,indices=0)
            world.step(render=False)
            q=robot.get_dof_positions(indices=0).numpy()[0]
            vel=robot.get_dof_velocities(indices=0).numpy()[0]
            bn=length*np.array([math.cos(q[pi]),math.sin(q[pi])])
            cn=bn+distal*np.array([math.cos(phi0+q[pi]+q[di]),math.sin(phi0+q[pi]+q[di])])
            residual=float(np.linalg.norm(cn-a)-rod)
            samples.append((t,goal,q[pi],q[di],expected(float(q[pi])),residual,*vel))
            if not np.isfinite(q).all() or abs(residual)>.001:
                abort="CONSTRAINT_DIVERGENCE";break
        data=np.asarray(samples)
        np.savez_compressed(args.output/"samples.npz",samples=data)
        result={"scope":"CPU_DISTANCE_JOINT_FOURBAR_MECHANISM_COUPON_NOT_FULL_HAND","abort":abort,"steps":len(data),"dof_names":names,
                "max_rod_length_error_m":float(np.max(np.abs(data[:,5]))),"max_follower_angle_error_rad":float(np.max(np.abs(data[:,3]-data[:,4]))),
                "actual_proximal_range_rad":[float(data[:,2].min()),float(data[:,2].max())],"wall_time_s":time.monotonic()-start,
                "reference_link_masses_kg":[.073035,.057879],"contact_or_object_truth_control":False,"state_written_after_reset":False,
                "source_measured_anchors":str(source),"self_lock_implemented":False}
        (args.output/"result.json").write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
    except Exception:
        (args.output/"error.txt").write_text(traceback.format_exc())
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        app.close()


if __name__=="__main__":
    main()
