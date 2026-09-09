"""Declared laboratory rotary/axial actuator with an independent joint wrench.

The original Nut stays a regular rigid body. A temporary external fixed joint
couples it to a small sensor tool on an actuator articulation. The apparatus is
never installed in a robot-assembly scene. No object truth enters its commands.
"""
import numpy as np


def author_instrumented_guide(stage,world,nut_path,initial_frame,*,torque_cap_nm=1.6,position_iterations=128):
    from pxr import UsdGeom,UsdPhysics,PhysxSchema,Gf
    from scipy.spatial.transform import Rotation
    from isaacsim.core.prims import SingleArticulation
    root='/World/InstrumentedModelDrive'
    if position_iterations not in (32,128):raise ValueError('only the baseline and declared convergence comparison are supported')
    if stage.GetPrimAtPath(root):raise ValueError('instrumented test apparatus already exists')
    UsdGeom.Xform.Define(stage,root)
    T=np.asarray(initial_frame);position=T[:3,3];q=Rotation.from_matrix(T[:3,:3]).as_quat()
    orient=Gf.Quatf(float(q[3]),Gf.Vec3f(*q[:3]))
    for name,mass in [('Base',.1),('Slider',.01),('Rotor',.01),('SensorTool',.001)]:
        x=UsdGeom.Xform.Define(stage,root+'/'+name);x.AddTranslateOp().Set(Gf.Vec3d(*position));x.AddOrientOp().Set(orient)
        body=x.GetPrim();UsdPhysics.RigidBodyAPI.Apply(body)
        m=UsdPhysics.MassAPI.Apply(body);m.CreateMassAttr(mass);m.CreateCenterOfMassAttr(Gf.Vec3f(0.))
        m.CreateDiagonalInertiaAttr(Gf.Vec3f(mass*.01**2/6))
        p=PhysxSchema.PhysxRigidBodyAPI.Apply(body);p.CreateDisableGravityAttr(True);p.CreateSleepThresholdAttr(0.)
        p.CreateLinearDampingAttr(0.);p.CreateAngularDampingAttr(0.)
        p.CreateSolverPositionIterationCountAttr(position_iterations);p.CreateSolverVelocityIterationCountAttr(1)
    anchor=UsdPhysics.FixedJoint.Define(stage,root+'/WorldAnchor')
    anchor.CreateBody1Rel().SetTargets([root+'/Base']);anchor.CreateLocalPos0Attr(Gf.Vec3f(*position));anchor.CreateLocalRot0Attr(orient)
    UsdPhysics.ArticulationRootAPI.Apply(anchor.GetPrim());art=PhysxSchema.PhysxArticulationAPI.Apply(anchor.GetPrim())
    art.CreateSolverPositionIterationCountAttr(position_iterations);art.CreateSolverVelocityIterationCountAttr(1);art.CreateSleepThresholdAttr(0.)
    slide=UsdPhysics.PrismaticJoint.Define(stage,root+'/AxialJoint')
    slide.CreateBody0Rel().SetTargets([root+'/Base']);slide.CreateBody1Rel().SetTargets([root+'/Slider']);slide.CreateAxisAttr('Z')
    slide.CreateLowerLimitAttr(-.02);slide.CreateUpperLimitAttr(.02)
    axial=UsdPhysics.DriveAPI.Apply(slide.GetPrim(),'linear');axial.CreateTypeAttr('force')
    axial.CreateStiffnessAttr(10000.);axial.CreateDampingAttr(20.);axial.CreateMaxForceAttr(3.0400615)
    axial.CreateTargetPositionAttr(0.);axial.CreateTargetVelocityAttr(0.)
    rotation=UsdPhysics.RevoluteJoint.Define(stage,root+'/RotaryJoint')
    rotation.CreateBody0Rel().SetTargets([root+'/Slider']);rotation.CreateBody1Rel().SetTargets([root+'/Rotor']);rotation.CreateAxisAttr('Z')
    drive=UsdPhysics.DriveAPI.Apply(rotation.GetPrim(),'angular');drive.CreateTypeAttr('force')
    drive.CreateStiffnessAttr(30.*np.pi/180.);drive.CreateDampingAttr(.1*np.pi/180.);drive.CreateMaxForceAttr(float(torque_cap_nm))
    drive.CreateTargetPositionAttr(0.);drive.CreateTargetVelocityAttr(0.)
    sensor=UsdPhysics.FixedJoint.Define(stage,root+'/TorqueSensor')
    sensor.CreateBody0Rel().SetTargets([root+'/Rotor']);sensor.CreateBody1Rel().SetTargets([root+'/SensorTool'])
    bridge=UsdPhysics.FixedJoint.Define(stage,root+'/DeclaredNutClamp')
    bridge.CreateBody0Rel().SetTargets([root+'/SensorTool']);bridge.CreateBody1Rel().SetTargets([nut_path])
    bridge.CreateExcludeFromArticulationAttr(True)
    tree=world.scene.add(SingleArticulation(root,name='instrumented_model_drive',reset_xform_properties=False))
    metadata=dict(scope='DECLARED_INSTRUMENTED_TEST_APPARATUS_NOT_ROBOT',nut_path=nut_path,
        source_nut_rigid_body_and_internal_joint_unchanged=True,external_clamp_path=str(bridge.GetPath()),
        sensor_path=str(sensor.GetPath()),source_payload_gravity_changed=False,
        apparatus_gravity_disabled=True,sensor_tool_mass_kg=.001,sensor_tool_inertia_kg_m2=.001*.01**2/6,
        sensor_semantics='External load on tool = negative native incoming joint wrench; retain raw signal. The small tool inertia remains in dynamic samples. No loaded tare.',
        torque_cap_nm=float(torque_cap_nm),axial_drive_cap_n=3.0400615)
    metadata['position_iterations']=position_iterations
    return bridge,drive,axial,{'tree':tree,'metadata':metadata}


def read_instrumented_guide(spec):
    tree=spec['tree'];view=tree._articulation_view
    row=view._metadata.joint_indices['TorqueSensor']+1
    raw=np.asarray(tree.get_measured_joint_forces())[row].astype(float)
    return dict(raw_incoming_wrench_sensor=raw.tolist(),external_wrench_sensor=(-raw).tolist(),
        external_axis_torque_nm=float(-raw[5]),loaded_tare_applied=False,
        actuator_joint_positions=np.asarray(tree.get_joint_positions()).astype(float).tolist(),
        actuator_dof_names=list(tree.dof_names))
