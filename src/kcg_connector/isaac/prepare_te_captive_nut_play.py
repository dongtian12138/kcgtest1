#!/usr/bin/env python3
"""Give the captive nut the bounded axial motion reported on the real part.

Only the internal Body/Nut bearing changes. It retains radial and tilt support,
allows rotation plus limited axial travel, and adds no connection to the world.
The reported approximately 1 mm travel is an estimate, not a measured tolerance.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from pxr import Sdf, Usd, UsdPhysics


def prepare(source_model, output, travel_m):
    if not 0 < travel_m <= .002:
        raise ValueError('Use a finite travel within the observed millimetre scale')
    source_model, output = Path(source_model).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Usd.Stage.Open(str(source_model));stage=Usd.Stage.Open(source.Flatten())
    path='/World/TE_J35FreeSplitPlug/Joints/CouplingNutRevolute'
    old=stage.GetPrimAtPath(path)
    if not old.IsA(UsdPhysics.RevoluteJoint):raise ValueError('Expected original captive-nut revolute')
    resistance=float(old.GetAttribute('physxJointAxis:angular:dynamicFrictionEffort').Get())
    # Keep the existing prim path and both local frames so callers and source
    # poses retain their identity; the old path name is not its new joint type.
    old.SetTypeName('PhysicsJoint')
    joint=UsdPhysics.Joint(old)
    joint.CreateExcludeFromArticulationAttr(True)
    for prop in list(old.GetProperties()):
        if prop.GetName().startswith(('drive:angular:', 'state:angular:', 'physxJointAxis:angular:')) or prop.GetName() in ('physics:axis','physics:lowerLimit','physics:upperLimit'):
            old.RemoveProperty(prop.GetName())
    schemas=old.GetMetadata('apiSchemas')
    tokens=list(schemas.GetAddedOrExplicitItems()) if schemas else []
    tokens=[t for t in tokens if not ('StateAPI' in t or t.endswith(':angular'))]
    old.SetMetadata('apiSchemas',Sdf.TokenListOp.CreateExplicit(tokens))
    for axis in ('transX','transY','rotX','rotY'):
        limit=UsdPhysics.LimitAPI.Apply(old,axis);limit.CreateLowAttr(1.);limit.CreateHighAttr(-1.)
    axial=UsdPhysics.LimitAPI.Apply(old,'transZ')
    axial.CreateLowAttr(-travel_m/2);axial.CreateHighAttr(travel_m/2)
    drive=UsdPhysics.DriveAPI.Apply(old,'rotZ')
    drive.CreateTypeAttr('force');drive.CreateStiffnessAttr(0.)
    drive.CreateDampingAttr(100.*np.pi/180.);drive.CreateMaxForceAttr(resistance)
    drive.CreateTargetVelocityAttr(0.)
    old.CreateAttribute('kcg:captiveNutAxialPlayM',Sdf.ValueTypeNames.Float).Set(travel_m)
    old.CreateAttribute('kcg:passiveResistanceNm',Sdf.ValueTypeNames.Float).Set(resistance)
    for body in ('Body','CouplingNut'):
        p=stage.GetPrimAtPath('/World/TE_J35FreeSplitPlug/'+body)
        schemas=p.GetMetadata('apiSchemas');tokens=list(schemas.GetAddedOrExplicitItems())
        tokens=[t for t in tokens if t not in ('PhysicsArticulationRootAPI','PhysxArticulationAPI')]
        p.SetMetadata('apiSchemas',Sdf.TokenListOp.CreateExplicit(tokens))
    stage.Flatten().Export(str(output/'connector_model.usdc'))
    check=Usd.Stage.Open(str(output/'connector_model.usdc'))
    for p in source.Traverse():
        if p.GetPath()==old.GetPath():continue
        q=check.GetPrimAtPath(p.GetPath())
        for prop in p.GetProperties():
            equal=(str(prop.Get())==str(q.GetAttribute(prop.GetName()).Get()) if isinstance(prop,Usd.Attribute)
                   else prop.GetTargets()==q.GetRelationship(prop.GetName()).GetTargets())
            if not equal:raise ValueError('Non-joint property changed: '+str(prop.GetPath()))
    report=json.loads(source_model.with_name('connector_model_assembly_scene.json').read_text())
    record={'scope':'CAPTIVE_NUT_BEARING_WITH_USER_REPORTED_AXIAL_TRAVEL',
            'source_model':str(source_model),'joint_path':path,'joint_type':'D6_WITH_FOUR_LOCKED_AXES',
            'axial_travel_m':travel_m,'axial_limits_m':[-travel_m/2,travel_m/2],
            'axial_datum':'Source CAD pose placed at mid-travel; exact end positions not measured',
            'travel_source':'User observed approximately 1 mm axial motion on real J599/26FJ35PN, 2026-09-10',
            'restoring_axial_spring_n_m':0.,'restoring_spring_status':'User confirmed the nut remains approximately where pushed; clearance motion, not a return spring',
            'passive_rotational_resistance_nm':resistance,'locked_relative_axes':['transX','transY','rotX','rotY'],
            'world_attachment_or_screw_kinematic_constraint':False,'original_geometry_material_mass_inertia_unchanged':True,
            'equivalent_thread_turn_for_full_axial_travel_deg':travel_m/.00762*360,
            'physical_validation':'PENDING_FIRST_TURN_COMPARISON'}
    report['captive_nut_axial_play']=record;report['passive_joint_representation']=record
    report['passive_joint_solver']={'representation':'MAXIMAL_COORDINATE_CAPTIVE_CYLINDRICAL_BEARING',
        'root_path':'/World/TE_J35FreeSplitPlug/Body','internal_joint_path':path,
        'floating_base':True,'articulation_root':False,'position_iterations':128,'velocity_iterations':1,
        'world_joint_added':False,'joint_degrees_of_freedom_changed':True,'mass_and_inertia_unchanged':True}
    report['joint_degrees_of_freedom_changed']=True
    report['native_joint_friction'].update(model='REGULARIZED_CAPPED_ZERO_SPEED_D6_TWIST_DRIVE',
        drive_axis='rotZ',maximum_resistance_nm=resistance,damping_nm_s_rad=100.,hardware_resistance_calibrated=False)
    (output/'connector_model_assembly_scene.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'preparation.json').write_text(json.dumps(record,indent=2)+'\n')
    adapter=Path(__file__).resolve().parents[3]/'artifacts/kcg_connector/model_delivery_20260908/package/install_model.py'
    shutil.copy2(adapter,output/'install_model.py')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--travel-m',type=float,default=.001)
    a=p.parse_args();prepare(a.source_model,a.output,a.travel_m)
