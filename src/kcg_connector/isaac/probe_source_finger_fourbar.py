"""Bounded CPU motion of one source finger with its measured internal rod.

Original source STL visuals, inertial tensors, and hinge frames are retained.
Optional protocols distinguish free motion, transmission self-lock, and contact
with an explicitly fixed laboratory nut. None is grasp/assembly acceptance.
Only the proximal hinge is driven after initialization.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def _origin(element):
    transform = np.eye(4)
    if element is not None:
        transform[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        transform[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
    return transform


def _turn(angle):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("z", angle).as_matrix()
    return transform


def _host(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def _binary_stl(path):
    """Read the unchanged binary STL triangles used by these source links."""
    record = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    with path.open("rb") as stream:
        header = stream.read(84)
        if len(header) != 84:
            raise ValueError(f"Truncated source STL: {path}")
        count = int.from_bytes(header[80:84], "little")
        if path.stat().st_size != 84 + 50 * count:
            raise ValueError(f"Expected the source binary STL without conversion: {path}")
        rows = np.fromfile(stream, dtype=record, count=count)
    return rows["vertices"].reshape(-1, 3).copy(), rows["normal"].copy()


def _target(time_s):
    low, high = math.radians(20.), math.radians(60.)
    if time_s < .5:
        return low, "initial_hold"
    if time_s < 1.75:
        u = (time_s - .5) / 1.25
        return low + (high-low) * (10*u**3-15*u**4+6*u**5), "closing"
    if time_s < 2.25:
        return high, "closed_hold"
    if time_s < 3.5:
        u = (time_s - 2.25) / 1.25
        return high - (high-low) * (10*u**3-15*u**4+6*u**5), "opening"
    return low, "final_hold"


def _worm_program(time_s, changing_load=False):
    if changing_load:
        if time_s<.2:return 0.,0.,"rest"
        if time_s<.6:return .15,0.,"moving_before_load"
        if time_s<1.2:return .15,-.3*(time_s-.6)/.6,"load_ramp_while_moving"
        if time_s<1.8:return .8,-.3,"closing_against_load"
        if time_s<2.4:return .8,-.3+.6*(time_s-1.8)/.6,"load_reversal_while_moving"
        if time_s<3.:return -.2,.3,"insufficient_opening_input"
        if time_s<3.6:return -.8,.3,"opening_against_load"
        return 0.,0.,"final_unloaded_hold"
    phases=(("rest",.2,0.,0.),("negative_load_hold",.4,0.,-.3),
            ("motor_closing_against_load",.4,1.2,-.3),("negative_hold_after_drive",.2,0.,-.3),
            ("motor_opening_with_load",.4,-.3,-.3),("unloaded_hold",.4,0.,0.),
            ("positive_load_hold",.4,0.,.3),("motor_opening_against_load",.4,-1.2,.3),
            ("positive_hold_after_drive",.2,0.,.3),("motor_closing_with_load",.4,.3,.3),
            ("final_unloaded_hold",.6,0.,0.))
    end=0.
    for name,duration,effort,load in phases:
        end+=duration
        if time_s<end-1e-12:return effort,load,name
    return 0.,0.,"final_unloaded_hold"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finger", choices=("f1", "f2", "f3"), default="f1")
    parser.add_argument("--wall-limit-s", type=float, default=180.)
    parser.add_argument("--no-images", action="store_true", help="Explicitly omit same-run Isaac camera evidence")
    parser.add_argument("--rod-method", choices=("distance","axial","tangent"), default="distance")
    parser.add_argument("--worm-port-test",action="store_true",help="Declared input-effort and external joint-load protocol; no grasp/assembly")
    parser.add_argument("--physics-hz",type=int,choices=(960,1920),default=960)
    parser.add_argument("--worm-integration",choices=("endpoint","midpoint_native","passive_split"),default="endpoint")
    parser.add_argument("--sensor-load-test",action="store_true",help="Known world force at a distal material point; verify calibrated moment about O")
    parser.add_argument("--changing-worm-load",action="store_true",help="Discriminating load-sign transition while the motor is active")
    parser.add_argument("--worm-robot-predictor",action="store_true",help="Predict friction branch using robot inertia and previously observed port response")
    parser.add_argument("--worm-zero-smoothing-nm",type=float,default=0.,help="Explicit numerical smoothing of load-sign crossing, not a calibrated friction parameter")
    parser.add_argument("--worm-friction-step",choices=("tangent","frozen_magnitude"),default="tangent")
    parser.add_argument("--nut-contact-test",action="store_true",help="Press, zero-input hold and active release against an explicitly fixed source nut")
    parser.add_argument("--contact-geometry",type=Path)
    parser.add_argument("--contact-drive-duration-s",type=float,default=1.8,help="Duration of each bounded press/release leg, with the same 0.6 Nm motor input")
    parser.add_argument("--nut-collider",choices=("visual-triangles","production-external-sdf"),default="visual-triangles")
    args = parser.parse_args()
    if args.worm_port_test and args.rod_method!="tangent":
        parser.error("Worm integration review requires the already tested tangent closure")
    if args.sensor_load_test and args.worm_port_test:
        parser.error("Sensor load and transmission integration are separate controlled comparisons")
    if args.changing_worm_load and not args.worm_port_test:
        parser.error("Changing load is a transmission test")
    if args.nut_contact_test and (not args.worm_port_test or args.contact_geometry is None or args.changing_worm_load):
        parser.error("Nut contact requires the worm, a measured-mechanism geometry plan, and no prescribed joint-load protocol")
    if not .5<=args.contact_drive_duration_s<=4.:
        parser.error("Contact drive duration must be bounded between 0.5 and 4 seconds")
    if not 0<=args.worm_zero_smoothing_nm<=.005:
        parser.error("Load-sign smoothing must stay within the declared small numerical range")
    if args.worm_integration=="passive_split" and (args.worm_robot_predictor or args.worm_zero_smoothing_nm):
        parser.error("Passive splitting uses no future-load predictor or friction smoothing")
    if not math.isfinite(args.wall_limit_s) or not 10 <= args.wall_limit_s <= 600:
        parser.error("wall limit must be finite and between10 and600 seconds")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    for name in (Path(__file__).name,"te_hand_fourbar.py","te_worm_drive.py"):
        shutil.copy2(Path(__file__).with_name(name),args.output/name)
    repo = Path(__file__).resolve().parents[3]
    contract_path = repo / "src/kcg_connector/config/hand_fourbar_20260912.json"
    urdf_path = repo / "src/iiwa_description/urdf/hand.xacro"
    from te_hand_fourbar import author_fourbar_rods, load_fourbar_contract
    contract, couplings = load_fourbar_contract(contract_path)
    cfg = contract["finger_joints"][args.finger]
    coupling = couplings[cfg["follower_joint"]]
    document = ET.parse(urdf_path).getroot()
    link_names = [cfg[k] for k in ("parent_link", "proximal_link", "distal_link")]
    joint_names = [cfg[k] for k in ("source_joint", "follower_joint")]
    links = {name: document.find(f"link[@name='{name}']") for name in link_names}
    joints = {name: document.find(f"joint[@name='{name}']") for name in joint_names}
    if any(v is None for v in [*links.values(), *joints.values()]):
        raise ValueError("Measured mechanism references missing source links/hinges")
    joint_frames = [_origin(joints[name].find("origin")) for name in joint_names]
    for name in joint_names:
        if not np.allclose(np.fromstring(joints[name].find("axis").get("xyz"), sep=" "), (0,0,1)):
            raise ValueError("This source probe expects the verified source+Z hinge axes")
    q_start = math.radians(20.)
    follower_start, _ = coupling.position_and_derivative(q_start)
    # Rigid placement of the whole coupon only: source local frames are unmodified.
    # Put the proximal root at0.15m and its finger plane vertical under gravity.
    beta = math.atan2(cfg["distal_joint_xy_m"][1], cfg["distal_joint_xy_m"][0])
    root_pose = np.eye(4)
    root_pose[:3, :3] = Rotation.from_euler("x", math.pi/2).as_matrix() @ Rotation.from_euler("z", math.pi/2-beta).as_matrix()
    root_pose[:3, 3] = (0., 0., .15)
    base_pose = root_pose @ np.linalg.inv(joint_frames[0])
    near_pose = base_pose @ joint_frames[0] @ _turn(q_start)
    tip_pose = near_pose @ joint_frames[1] @ _turn(follower_start)
    initial_poses = [base_pose, near_pose, tip_pose]
    fixture_pose = geometry = None
    if args.nut_contact_test:
        from kcg_connector.grasp.carts_v2.models import load_v2_inputs
        geometry = json.loads(args.contact_geometry.read_text())
        if geometry["finger_mechanism_id"] != contract["mechanism_id"] or geometry["hand_variant"] != "LEGACY_NAIL_PRESENT":
            raise ValueError("Contact placement must use the same measured nail-present mechanism")
        inputs = load_v2_inputs(repo,config_path=geometry["source_config"],object_id="te_deutsch_d38999_26fj35pn_step",
                               finger_mechanism_path=contract_path)
        full_q = np.r_[np.zeros(7),geometry["first_contact_hand_positions_rad"]]
        fk = inputs.robot_model.forward_kinematics(tuple(full_q),enforce_limits=False)
        fixture_pose = (base_pose @ np.linalg.inv(fk[cfg["parent_link"]]) @ fk["handbase_link"]
                        @ np.linalg.inv(geometry["canonical_body_from_hand_for_nut_grasp"]))
        shutil.copy2(args.contact_geometry,args.output/"contact_geometry.json")
    samples, frames = [], []
    sample_stream = None
    app = None
    start = time.monotonic()
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": True, "hide_ui": True})
        import omni.usd
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt
        from isaacsim.core.api import World
        from isaacsim.core.experimental.prims import Articulation, RigidPrim
        from isaacsim.core.simulation_manager import SimulationManager
        SimulationManager.set_physics_sim_device("cpu")
        dt = 1 / args.physics_hz
        world = World(stage_units_in_meters=1., physics_dt=dt, rendering_dt=1/60., backend="numpy", device="cpu")
        stage = omni.usd.get_context().get_stage()
        scene = UsdPhysics.Scene(stage.GetPrimAtPath(world.get_physics_context().prim_path))
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0., 0., -1.))
        scene.CreateGravityMagnitudeAttr(9.81)
        physics = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        physics.CreateEnableGPUDynamicsAttr(False)
        physics.CreateBroadphaseTypeAttr("MBP")
        physics.CreateSolverTypeAttr("TGS")
        physics.CreateEnableExternalForcesEveryIterationAttr(True)
        for minimum, maximum, value in ((physics.CreateMinPositionIterationCountAttr, physics.CreateMaxPositionIterationCountAttr,64),
                                         (physics.CreateMinVelocityIterationCountAttr, physics.CreateMaxVelocityIterationCountAttr,4)):
            minimum(value); maximum(value)
        root_path = "/World/SourceFinger"
        root = UsdGeom.Xform.Define(stage, root_path)
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
        articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
        articulation_api.CreateSolverPositionIterationCountAttr(64)
        articulation_api.CreateSolverVelocityIterationCountAttr(4)

        def quaternion(matrix):
            x, y, z, w = Rotation.from_matrix(matrix).as_quat()
            return Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))

        body_paths, source_records = [], []
        for index, (name, pose) in enumerate(zip(link_names, initial_poses)):
            link = links[name]
            path = root_path + "/" + name
            body_paths.append(path)
            body = UsdGeom.Xform.Define(stage, path)
            body.AddTranslateOp().Set(Gf.Vec3d(*pose[:3,3]))
            body.AddOrientOp().Set(quaternion(pose[:3,:3]))
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            inertial = link.find("inertial")
            inertial_pose = _origin(inertial.find("origin"))
            source_mass = float(inertial.find("mass").get("value"))
            row = inertial.find("inertia")
            tensor = np.array([[float(row.get("ixx")),float(row.get("ixy")),float(row.get("ixz"))],
                               [float(row.get("ixy")),float(row.get("iyy")),float(row.get("iyz"))],
                               [float(row.get("ixz")),float(row.get("iyz")),float(row.get("izz"))]])
            tensor_link = inertial_pose[:3,:3] @ tensor @ inertial_pose[:3,:3].T
            diagonal, principal = np.linalg.eigh(tensor_link)
            if source_mass <= 0 or np.any(diagonal <= 0):
                raise ValueError("Source inertia is not positive definite")
            if np.linalg.det(principal) < 0:
                principal[:,0] *= -1
            mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
            mass.CreateMassAttr(source_mass)
            mass.CreateCenterOfMassAttr(Gf.Vec3f(*inertial_pose[:3,3]))
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*diagonal))
            mass.CreatePrincipalAxesAttr(quaternion(principal))
            mesh_records = []
            for vi, visual in enumerate(link.findall("visual")):
                reference = visual.find("geometry/mesh")
                if reference is None:
                    raise ValueError("Expected original STL source visual")
                filename = reference.get("filename")
                prefix = "package://iiwa_description/"
                if not filename.startswith(prefix):
                    raise ValueError("Unexpected source mesh package")
                mesh_path = repo / "src/iiwa_description" / filename[len(prefix):]
                points, normals = _binary_stl(mesh_path)
                scale = np.fromstring(reference.get("scale", "1 1 1"), sep=" ")
                points *= scale.astype(np.float32)
                mesh = UsdGeom.Mesh.Define(stage, path + f"/SourceVisual{vi}")
                mesh.AddTransformOp().Set(Gf.Matrix4d(*_origin(visual.find("origin")).T.ravel().tolist()))
                mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
                mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(normals),3,dtype=np.int32)))
                mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.arange(len(points),dtype=np.int32)))
                mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
                mesh.SetNormalsInterpolation(UsdGeom.Tokens.uniform)
                mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
                mesh.CreateDoubleSidedAttr(True)
                material_color = visual.find("material/color")
                color = np.fromstring(material_color.get("rgba"), sep=" ")[:3] if material_color is not None else np.array([.8,.8,.8])
                display_color=((.16,.23,.32),(.12,.36,.58),(.55,.30,.06))[index]
                mesh.CreateDisplayColorAttr([Gf.Vec3f(*display_color)])
                mesh_records.append({"path":str(mesh_path),"triangles":len(normals),"source_scale":scale.tolist(),
                                     "original_display_color":color.tolist(),"diagnostic_display_color":display_color})
            source_records.append({"link":name,"mass_kg":source_mass,"center_of_mass_local_m":inertial_pose[:3,3].tolist(),
                                   "inertia_tensor_link_kg_m2":tensor_link.tolist(),"principal_diagonal_kg_m2":diagonal.tolist(),
                                   "principal_axes_xyzw":Rotation.from_matrix(principal).as_quat().tolist(),"visuals":mesh_records,
                                   "initial_world_pose":pose.tolist()})

        fixed = UsdPhysics.FixedJoint.Define(stage, root_path + "/FixedSourceBase")
        fixed.CreateBody1Rel().SetTargets([Sdf.Path(body_paths[0])])
        # Its world-side frame equals the declared base pose; reset need not move the base.
        fixed.CreateLocalPos0Attr(Gf.Vec3f(*base_pose[:3,3]))
        fixed.CreateLocalRot0Attr(quaternion(base_pose[:3,:3]))
        fixed.CreateLocalPos1Attr(Gf.Vec3f(0.))
        fixed.CreateLocalRot1Attr(Gf.Quatf(1.))
        source_joint_records = []
        for i, (name, frame, q_initial) in enumerate(zip(joint_names,joint_frames,(q_start,follower_start))):
            j = UsdPhysics.RevoluteJoint.Define(stage, root_path + "/" + name)
            j.CreateBody0Rel().SetTargets([Sdf.Path(body_paths[i])])
            j.CreateBody1Rel().SetTargets([Sdf.Path(body_paths[i+1])])
            j.CreateAxisAttr("Z")
            j.CreateLocalPos0Attr(Gf.Vec3f(*frame[:3,3]))
            j.CreateLocalRot0Attr(quaternion(frame[:3,:3]))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.))
            j.CreateLocalRot1Attr(Gf.Quatf(1.))
            j.CreateCollisionEnabledAttr(False)
            limit = joints[name].find("limit")
            lower, upper, speed = (float(limit.get(k)) for k in ("lower","upper","velocity"))
            j.CreateLowerLimitAttr(math.degrees(lower)); j.CreateUpperLimitAttr(math.degrees(upper))
            PhysxSchema.PhysxJointAPI.Apply(j.GetPrim()).CreateMaxJointVelocityAttr(math.degrees(speed))
            state = PhysxSchema.JointStateAPI.Apply(j.GetPrim(), "angular")
            state.CreatePositionAttr(math.degrees(q_initial)); state.CreateVelocityAttr(0.)
            drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(math.radians(120.) if i==0 else 0.)
            drive.CreateDampingAttr(math.radians(2.) if i==0 else 0.)
            drive.CreateMaxForceAttr(3.5 if i==0 else 0.)
            drive.CreateTargetPositionAttr(math.degrees(q_initial))
            drive.CreateTargetVelocityAttr(0.)
            dynamics = joints[name].find("dynamics")
            source_joint_records.append({"joint":name,"source_parent_from_joint":frame.tolist(),"initial_angle_rad":q_initial,
                                         "source_limits_rad":[lower,upper],"source_velocity_limit_rad_s":speed,
                                         "source_urdf_dynamics":dict(dynamics.attrib) if dynamics is not None else {},
                                         "additional_passive_joint_friction_or_damping_authored":False})
        rod_record = author_fourbar_rods(stage, root_path, contract_path, finger_names=[args.finger],representation=args.rod_method)
        contact_view = None
        if args.nut_contact_test:
            import carb
            from build_te_free_split_plug import _load_single_usd_mesh, NUT_VISUAL
            carb.settings.get_settings().set("/physics/disableContactProcessing",False)
            material=UsdShade.Material.Define(stage,"/World/ContactReferenceMaterial")
            mat=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            mat.CreateStaticFrictionAttr(.45);mat.CreateDynamicFrictionAttr(.45);mat.CreateRestitutionAttr(0.)
            PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim()).CreateFrictionCombineModeAttr("average")
            fixture_path="/World/DeclaredFixedNut"
            fixture=UsdGeom.Mesh.Define(stage,fixture_path)
            fixture.AddTransformOp().Set(Gf.Matrix4d(*fixture_pose.T.ravel().tolist()))
            vertices,faces=_load_single_usd_mesh(NUT_VISUAL)
            collision_source=str(NUT_VISUAL)
            if args.nut_collider=="production-external-sdf":
                collision_source=str(repo/"artifacts/kcg_connector/te_connector_contact_repaired_20260911/connector_model.usdc")
                source_stage=Usd.Stage.Open(collision_source)
                source_mesh=UsdGeom.Mesh(source_stage.GetPrimAtPath("/World/TE_J35FreeSplitPlug/CouplingNut/ExternalSurfaceContact"))
                if not source_mesh or UsdGeom.Xformable(source_mesh.GetPrim()).GetOrderedXformOps():
                    raise ValueError("Expected the existing external Nut contact mesh in its source body frame")
                vertices=np.asarray(source_mesh.GetPointsAttr().Get(),np.float32)
                faces=np.asarray(source_mesh.GetFaceVertexIndicesAttr().Get(),np.int32).reshape(-1,3)
            fixture.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
            fixture.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces),3,np.int32)))
            fixture.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.ravel()))
            fixture.CreateSubdivisionSchemeAttr("none");fixture.CreateDisplayColorAttr([Gf.Vec3f(.42,.48,.53)])
            tip_mesh=stage.GetPrimAtPath(body_paths[2]+"/SourceVisual0")
            for prim,approximation in ((fixture.GetPrim(),"sdf" if args.nut_collider=="production-external-sdf" else "none"),(tip_mesh,"sdf")):
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)
                col=PhysxSchema.PhysxCollisionAPI.Apply(prim)
                col.CreateContactOffsetAttr(.00005);col.CreateRestOffsetAttr(0.)
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,materialPurpose="physics")
                if approximation=="sdf":
                    sdf=PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
                    sdf.CreateSdfResolutionAttr(1024);sdf.CreateSdfSubgridResolutionAttr(6);sdf.CreateSdfNarrowBandThicknessAttr(.002)
                    sdf.CreateSdfTriangleCountReductionFactorAttr(1.)
                    PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(prim).CreateWeldToleranceAttr(0.)
            sdf=PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(tip_mesh)
            sdf.CreateSdfResolutionAttr(1024);sdf.CreateSdfSubgridResolutionAttr(6);sdf.CreateSdfNarrowBandThicknessAttr(.002)
            PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(tip_mesh).CreateWeldToleranceAttr(0.)
            PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(body_paths[2])).CreateThresholdAttr(0.)
            contact_view=RigidPrim([body_paths[2]],contact_filter_paths=[fixture_path],max_contact_count=2048)
            (args.output/"contact_fixture.json").write_text(json.dumps({
                "scope":"DECLARED_FIXED_LABORATORY_NUT_SINGLE_FINGER_ONLY","fixture_is_static":True,
                "source_nut":str(NUT_VISUAL),"nut_collider":args.nut_collider,"collision_source":collision_source,
                "fixture_world_from_source_body":fixture_pose.tolist(),
                "distal_source_geometry_unchanged":True,"distal_sdf_resolution":1024,
                "contact_offset_m":.00005,"rest_offset_m":0.,"friction_development_reference":.45,
                "colliders_enabled_only_on_terminal_link_and_fixture":True,
                "contact_truth_is_recorded_for_offline_review_only":True},indent=2))
        (args.output/"source_authoring.json").write_text(json.dumps({"source_urdf":str(urdf_path),"finger":args.finger,
            "links":source_records,"hinges":source_joint_records,"rod":rod_record,"gravity_m_s2":[0,0,-9.81],
            "physics_hz":args.physics_hz,"position_iterations":64,"velocity_iterations":4,
            "contact_colliders_authored":bool(args.nut_contact_test),"original_source_files_modified":False,
            "worm_self_lock_implemented":bool(args.worm_port_test),"scope":"SOURCE_SINGLE_FINGER_DIAGNOSTIC"},indent=2))

        rgb = product = None
        if not args.no_images:
            import omni.replicator.core as rep
            camera_path = "/World/SourceFingerEvidenceCamera"
            camera = UsdGeom.Camera.Define(stage, camera_path)
            eye, focus = Gf.Vec3d(.11,-.43,.29), Gf.Vec3d(-.045,0.,.155)
            camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(eye,focus,Gf.Vec3d(0,0,1)).GetInverse())
            camera.CreateFocalLengthAttr(36.)
            camera.CreateHorizontalApertureAttr(36.)
            camera.CreateVerticalApertureAttr(27.)
            camera.CreateClippingRangeAttr(Gf.Vec2f(.01,3.))
            UsdLux.DomeLight.Define(stage,"/World/SourceFingerEvidenceLight").CreateIntensityAttr(450.)
            product = rep.create.render_product(camera_path,(800,600))
            rgb = rep.AnnotatorRegistry.get_annotator("rgb"); rgb.attach([product.path])
        stage.GetRootLayer().Export(str(args.output/"scene.usdc"))
        bodies = RigidPrim(body_paths,resolve_paths=False)
        world.reset()
        robot = Articulation(root_path)
        names = list(robot.dof_names)
        if set(names) != set(joint_names):
            raise ValueError(f"Unexpected source finger DOFs: {names}")
        pi, di = (names.index(n) for n in joint_names)
        robot.set_dof_gains(np.array([120. if n==joint_names[0] else 0. for n in names]),
                           np.array([2. if n==joint_names[0] else 0. for n in names]),indices=0)
        robot.set_dof_max_efforts(np.array([3.5 if n==joint_names[0] else 0. for n in names]),indices=0)
        worm=None
        if args.worm_port_test:
            from te_worm_drive import WormDrive,WormReference
            reference=(WormReference(transmission_damping=0.,output_viscosity=2.) if args.worm_integration=="passive_split" else
                       WormReference(load_zero_smoothing=args.worm_zero_smoothing_nm))
            worm=WormDrive(float(_host(robot.get_dof_positions(indices=0))[0,pi]),integration=args.worm_integration,
                           reference=reference,
                           friction_step=args.worm_friction_step)
            (args.output/"worm_reference.json").write_text(json.dumps({
                "parameters":vars(worm.reference),"input_effort_boundary":worm.reference.input_effort_boundary,
                "scope":"UNCALIBRATED_TWO_PORT_SELF_LOCK_HYPOTHESIS","native_passive_friction":0.,"integration":args.worm_integration,
                "friction_step":"exact_implicit_motor_inclusion" if args.worm_integration=="passive_split" else args.worm_friction_step,
                "output_reference_not_motor_rating":True,"input_state_is_not_actual_joint_position":True},indent=2))
        # No pose/position/velocity setters occur after reset.
        start = time.monotonic()
        initial_time = float(world.current_time)
        sample_stream = (args.output/"samples.jsonl").open("w")
        abort = None
        image_error = None
        duration=(1.2+2*args.contact_drive_duration_s) if args.nut_contact_test else 4.
        total_steps=round(duration/dt)
        capture_steps = {0,*[round(t/dt)-1 for t in (.5,1.75,2.25,3.5,4.)]}
        if args.nut_contact_test:
            close_end=.2+args.contact_drive_duration_s
            capture_steps = {0,*[round(t/dt)-1 for t in (.8,close_end-.1,close_end+.5,duration-.6,duration)]}
        for step in range(total_steps):
            if time.monotonic()-start > args.wall_limit_s:
                abort="WALL_LIMIT"; break
            command_time = step*dt
            goal, phase = _target(command_time)
            fixture_force=np.zeros(3)
            if args.sensor_load_test:
                goal=math.radians(20.)
                if command_time<.5:phase="sensor_tare"
                elif command_time<1.5:phase="positive_force";fixture_force[0]=5.
                elif command_time<2.:phase="unloaded"
                elif command_time<3.:phase="negative_force";fixture_force[0]=-5.
                else:phase="final_unloaded"
            before_q=_host(robot.get_dof_positions(indices=0))[0]
            before_velocity=_host(robot.get_dof_velocities(indices=0))[0]
            targets = np.zeros((1,len(names)))
            load=0.
            if worm is None:
                targets[0,pi] = goal
            else:
                input_effort,load,phase=_worm_program(command_time,args.changing_worm_load)
                if args.nut_contact_test:
                    load=0.
                    if command_time<.2:input_effort=0.;phase="contact_initial_rest"
                    elif command_time<close_end:input_effort=.6;phase="motor_pressing_nut"
                    elif command_time<close_end+.6:input_effort=0.;phase="contact_zero_input_hold"
                    elif command_time<duration-.4:
                        input_effort=-.6 if worm.input_angle>q_start else 0.
                        phase="motor_active_release" if input_effort else "release_position_reached"
                    else:input_effort=0.;phase="released_zero_input_hold"
                prediction={}
                if args.worm_robot_predictor:
                    mapping=np.zeros(len(names));mapping[pi]=1.;mapping[di]=coupling.position_and_derivative(float(before_q[pi]))[1]
                    inertia=float(mapping@_host(robot.get_mass_matrices(indices=0))[0]@mapping)
                    if worm.previous_output_velocity is None or worm.last_observed_effort is None:
                        load_prediction=-float(mapping@_host(robot.get_dof_gravity_compensation_forces(indices=0))[0])
                    else:
                        load_prediction=inertia*(float(before_velocity[pi])-worm.previous_output_velocity)/dt-float(worm.last_observed_effort)
                    prediction={"predictor_inertia":inertia,"predictor_load":load_prediction}
                wd=worm.prepare(float(before_q[pi]),float(before_velocity[pi]),input_effort,dt,**prediction)
                targets[0,pi]=wd["position_target"]
                kp=np.zeros(len(names));kd=np.zeros(len(names))
                kp[pi]=wd["stiffness"];kd[pi]=wd["damping"]
                robot.set_dof_gains(kp,kd,indices=0)
                external=np.zeros((1,len(names)));external[0,pi]=load
                robot.set_dof_efforts(external,indices=0)
            robot.set_dof_position_targets(targets,indices=0)
            if args.sensor_load_test:
                load_positions,load_orientations=bodies.get_world_poses()
                load_positions,load_orientations=_host(load_positions),_host(load_orientations)
                load_rotation=Rotation.from_quat(load_orientations[2,[1,2,3,0]]).as_matrix()
                load_point=load_positions[2]+load_rotation@np.asarray(cfg["distal_anchor_local_m"])
                bodies.apply_forces_and_torques_at_pos(forces=fixture_force[None,:],positions=load_point[None,:],indices=[2],local_frame=False)
            if args.rod_method=="tangent":
                from te_hand_fourbar import update_fourbar_tangents
                update_fourbar_tangents(stage,root_path,couplings,{joint_names[0]:float(before_q[pi])})
            world.step(render=False)
            q = _host(robot.get_dof_positions(indices=0))[0]
            velocity = _host(robot.get_dof_velocities(indices=0))[0]
            positions, orientations = bodies.get_world_poses()
            positions, orientations = _host(positions), _host(orientations)
            if not np.isfinite(np.r_[q,velocity,positions.ravel(),orientations.ravel()]).all():
                abort="NONFINITE_PHYSICAL_STATE"; break
            expected, derivative = coupling.position_and_derivative(float(q[pi]))
            body_rotations = Rotation.from_quat(orientations[:,[1,2,3,0]]).as_matrix()
            mass_matrix=_host(robot.get_mass_matrices(indices=0))[0]
            kinetic=float(.5*velocity@mass_matrix@velocity)
            potential=sum(r["mass_kg"]*9.81*float((positions[j]+body_rotations[j]@np.asarray(r["center_of_mass_local_m"]))[2])
                          for j,r in enumerate(source_records))
            a_world = positions[0] + body_rotations[0] @ np.asarray(cfg["base_anchor_parent_local_m"])
            c_world = positions[2] + body_rotations[2] @ np.asarray(cfg["distal_anchor_local_m"])
            length_error = float(np.linalg.norm(c_world-a_world)-cfg["rod_length_m"])
            follower_error = math.atan2(math.sin(q[di]-expected),math.cos(q[di]-expected))
            record={"step":step,"physics_time_s":float(world.current_time),"elapsed_physics_time_s":float(world.current_time)-initial_time,
                    "command_time_s":command_time,"phase":phase,"target_proximal_rad":float(targets[0,pi]),
                    "q_proximal_rad":float(q[pi]),"q_distal_rad":float(q[di]),
                    "dq_proximal_rad_s":float(velocity[pi]),"dq_distal_rad_s":float(velocity[di]),
                    "expected_distal_rad":expected,"coupling_derivative":derivative,
                    "follower_angle_error_rad":follower_error,"rod_length_error_from_body_poses_m":length_error,
                    "kinetic_energy_j":kinetic,"gravity_potential_energy_j":potential,
                    "applied_external_joint_effort_nm":load,
                    "rod_length_error_from_joint_angles_m":coupling.closure_error(float(q[pi]),float(q[di])),
                    "body_positions_world_m":positions.tolist(),"body_orientations_wxyz":orientations.tolist(),
                    "A_world_m":a_world.tolist(),"C_world_m":c_world.tolist()}
            if args.sensor_load_test:
                projected=_host(robot.get_dof_projected_joint_forces(indices=0))[0]
                joint_origin=positions[0]+body_rotations[0]@joint_frames[0][:3,3]
                axis_world=body_rotations[0]@joint_frames[0][:3,:3]@np.array([0.,0.,1.])
                external_moment=float(np.cross(load_point-joint_origin,fixture_force)@axis_world)
                gravity_moment=0.
                for j in (1,2):
                    com=positions[j]+body_rotations[j]@np.asarray(source_records[j]["center_of_mass_local_m"])
                    gravity_moment+=float(np.cross(com-joint_origin,[0.,0.,-9.81*source_records[j]["mass_kg"]])@axis_world)
                record.update(sensor_fixture_force_world_n=fixture_force.tolist(),sensor_fixture_point_world_m=load_point.tolist(),
                              expected_external_moment_about_O_nm=external_moment,
                              expected_holding_reaction_about_O_nm=-external_moment-gravity_moment,
                              proximal_projected_reaction_nm=float(projected[pi]),
                              distal_projected_reaction_nm=float(projected[di]),
                              ideal_gravity_compensated_sensor_nm=float(projected[pi])+gravity_moment,
                              actuator_generalized_effort_nm=float(projected[pi]+derivative*projected[di]))
            if worm is not None:
                projected=_host(robot.get_dof_projected_joint_forces(indices=0))[0]
                native_slope=coupling.position_and_derivative(float(before_q[pi]))[1]
                observed=float(projected[pi]+native_slope*projected[di]-load)
                wr=worm.complete(float(q[pi]),float(velocity[pi]),observed_drive_effort=observed)
                record.update({"worm_"+k:v for k,v in wr.items()})
                record["projected_joint_efforts_nm"]=projected.tolist()
                record["transmission_spring_energy_j"]=.5*worm.reference.transmission_stiffness*(worm.input_angle-float(q[pi]))**2
            if contact_view is not None:
                forces,points,normals,distances,counts,starts=(_host(x) for x in contact_view.get_contact_force_data(dt=dt))
                count=int(counts.ravel()[0]);offset=int(starts.ravel()[0]);sl=slice(offset,offset+count)
                record.update(contact_count=count,contact_normal_force_sum_n=float(forces[sl].sum()),
                              contact_normal_forces_n=forces[sl].ravel().tolist(),
                              contact_points_world_m=points[sl].tolist(),contact_normals_world=normals[sl].tolist(),
                              contact_separations_m=distances[sl].ravel().tolist())
            samples.append(record); sample_stream.write(json.dumps(record)+"\n")
            if step%240==0:
                sample_stream.flush()
                (args.output/"progress.json").write_text(json.dumps({**{k:record[k] for k in ("step","phase","elapsed_physics_time_s","q_proximal_rad","q_distal_rad","rod_length_error_from_body_poses_m")},"wall_time_s":time.monotonic()-start}))
            if abs(length_error)>.002 or abs(follower_error)>.05:
                abort="FOURBAR_CONSTRAINT_DIVERGENCE"; break
            if worm is not None and (not wr["predicted_branch_consistent"] or wr["drive_saturation"] or
                                     wr.get("elastic_effort_boundary_exceeded",False) or wr["friction_heat_j"] < -1e-9):
                abort="WORM_PORT_BRANCH_OR_EFFORT_BOUNDARY";break
            if rgb is not None and step in capture_steps:
                # Freeze simulation while rendering the same physical state. The
                # rendered source meshes are evidence of this run, not replay.
                import carb
                import omni.kit.app
                import omni.timeline
                from PIL import Image
                before_time=float(world.current_time)
                world.pause()
                timeline=omni.timeline.get_timeline_interface(); auto=timeline.is_auto_updating()
                settings=carb.settings.get_settings(); play=settings.get("/app/player/playSimulations")
                try:
                    timeline.set_auto_update(False); timeline.commit_silently()
                    settings.set("/app/player/playSimulations",False)
                    for _ in range(3):
                        omni.kit.app.get_app().update()
                    settings.set("/app/player/playSimulations",play)
                    rep.orchestrator.step(rt_subframes=1,delta_time=0.,pause_timeline=False)
                    frame=np.asarray(rgb.get_data()).copy()
                    if frame.ndim==3 and frame.shape[:2]==(600,800):
                        path=args.output/f"source_finger_{step:04d}.png"
                        Image.fromarray(frame[:,:,:3].astype(np.uint8)).save(path)
                        frames.append({"step":step,"physics_time_s":before_time,"path":str(path),"identity":"SAME_RUN_ISAAC_CAMERA"})
                    else:
                        image_error=f"Empty/invalid RGB readback at step{step}: {frame.shape}"
                except Exception:
                    image_error=traceback.format_exc()
                    (args.output/"camera_error.txt").write_text(image_error)
                    rgb=None
                finally:
                    settings.set("/app/player/playSimulations",play)
                    timeline.set_auto_update(auto); timeline.commit_silently(); world.play()
                if abs(float(world.current_time)-before_time)>1e-9:
                    raise RuntimeError("Evidence capture advanced physical time")
        sample_stream.flush(); sample_stream.close(); sample_stream=None
        (args.output/"frames.json").write_text(json.dumps(frames,indent=2))
        if not samples:
            raise RuntimeError("Source finger produced no physical samples")
        scalar_fields=[k for k,v in samples[0].items() if isinstance(v,(int,float))]
        np.savez_compressed(args.output/"samples.npz",scalar_columns=np.asarray(scalar_fields),
                            scalar_data=np.asarray([[r[k] for k in scalar_fields] for r in samples]),
                            body_positions_world_m=np.asarray([r["body_positions_world_m"] for r in samples]),
                            body_orientations_wxyz=np.asarray([r["body_orientations_wxyz"] for r in samples]),
                            body_names=np.asarray(link_names))
        last_hold=[r for r in samples if r["phase"]=="final_hold"]
        result={"scope":"ORIGINAL_SOURCE_SINGLE_FINGER_DIAGNOSTIC_NOT_GRASP_OR_ASSEMBLY", "finger":args.finger,
                "rod_method":args.rod_method,
                "abort":abort,"completed_scheduled_motion":abort is None and len(samples)==total_steps,"steps":len(samples),
                "dof_names":names,"body_names":link_names,"wall_time_s":time.monotonic()-start,
                "max_rod_length_error_from_body_poses_m":max(abs(r["rod_length_error_from_body_poses_m"]) for r in samples),
                "max_rod_length_error_from_joint_angles_m":max(abs(r["rod_length_error_from_joint_angles_m"]) for r in samples),
                "max_follower_angle_error_rad":max(abs(r["follower_angle_error_rad"]) for r in samples),
                "max_proximal_tracking_error_rad":max(abs(r["q_proximal_rad"]-r["target_proximal_rad"]) for r in samples) if worm is None else None,
                "actual_proximal_range_rad":[min(r["q_proximal_rad"] for r in samples),max(r["q_proximal_rad"] for r in samples)],
                "final_hold_mean_positions_rad":([float(np.mean([r[k] for r in last_hold])) for k in ("q_proximal_rad","q_distal_rad")] if last_hold else None),
                "gravity_m_s2":[0,0,-9.81],"proximal_drive":{"stiffness_nm_rad":120.,"damping_nm_s_rad":2.,"effort_cap_nm":3.5},
                "distal_drive_zero":True,"state_written_after_reset":False,"self_lock_implemented":worm is not None,
                "drive_law":"EXPERIMENTAL_TWO_PORT_WORM" if worm is not None else "OUTPUT_PD",
                "worm_integration":args.worm_integration if worm is not None else None,
                "worm_friction_step":("exact_implicit_motor_inclusion" if args.worm_integration=="passive_split" else
                                      args.worm_friction_step) if worm is not None else None,
                "sensor_load_protocol":bool(args.sensor_load_test),
                "changing_worm_load":bool(args.changing_worm_load),
                "worm_robot_predictor":bool(args.worm_robot_predictor),
                "physics_hz":args.physics_hz,
                "contact_drive_duration_s":args.contact_drive_duration_s if args.nut_contact_test else None,
                "external_contacts_tested":any(r.get("contact_normal_force_sum_n",0)>0 for r in samples),
                "maximum_contact_normal_force_sum_n":max(r.get("contact_normal_force_sum_n",0) for r in samples),
                "colliders_authored":bool(args.nut_contact_test),"hardware_authorized":False,
                "same_run_camera_frames":len(frames),"camera_requested":not args.no_images,"camera_error":image_error,
                "original_source_files_modified":False,"rod_contract":str(contract_path)}
        (args.output/"result.json").write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
    except Exception:
        error=traceback.format_exc()
        (args.output/"error.txt").write_text(error)
        (args.output/"failure_context.json").write_text(json.dumps({"recorded_samples":len(samples),"last_sample":samples[-1] if samples else None,"same_run_frames":frames},indent=2))
        print(error,flush=True)
        raise
    finally:
        if sample_stream is not None:
            sample_stream.flush(); sample_stream.close()
        if app is not None:
            app.close()


if __name__=="__main__":
    main()
