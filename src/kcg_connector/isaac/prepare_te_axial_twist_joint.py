#!/usr/bin/env python3
"""Represent the captive nut's continuous rotation with the D6 twist axis.

PhysX twist is joint X. Body-local Z remains the same physical shaft axis;
only joint coordinates change, with identical physical degrees of freedom.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from pxr import Gf, Sdf, Usd, UsdPhysics


def update_installer_axis_validation(path):
    """Accept both representations while checking the same physical shaft."""
    path = Path(path)
    text = path.read_text()
    if 'rotation_coordinate=prim.GetAttribute' in text:
        return
    needle = """        for axis in ('transX','transY','rotX','rotY'):
            limit=UsdPhysics.LimitAPI(prim,axis)
            if limit.GetLowAttr().Get()!=1. or limit.GetHighAttr().Get()!=-1.:return False
        axial=UsdPhysics.LimitAPI(prim,'transZ')"""
    replacement = """        rotation_coordinate=prim.GetAttribute('kcg:rotationCoordinate').Get() or 'rotZ'
        axial_coordinate=prim.GetAttribute('kcg:axialCoordinate').Get() or 'transZ'
        if (rotation_coordinate,axial_coordinate) not in (('rotZ','transZ'),('rotX','transX')):return False
        locked=(('transY','transZ','rotY','rotZ') if rotation_coordinate=='rotX' else ('transX','transY','rotX','rotY'))
        for axis in locked:
            limit=UsdPhysics.LimitAPI(prim,axis)
            if limit.GetLowAttr().Get()!=1. or limit.GetHighAttr().Get()!=-1.:return False
        if rotation_coordinate=='rotX':
            joint=UsdPhysics.Joint(prim)
            for q in (joint.GetLocalRot0Attr().Get(),joint.GetLocalRot1Attr().Get()):
                physical_axis=Gf.Rotation(Gf.Quatd(q)).TransformDir(Gf.Vec3d(1.,0.,0.))
                if (physical_axis-Gf.Vec3d(0.,0.,1.)).GetLength()>1e-6:return False
        axial=UsdPhysics.LimitAPI(prim,axial_coordinate)"""
    if needle not in text:
        raise ValueError('Installer bearing validation changed; review before patching')
    text = text.replace('from pxr import Sdf,', 'from pxr import Gf, Sdf,').replace(needle, replacement)
    path.write_text(text)


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    original = Usd.Stage.Open(str(source)); stage = Usd.Stage.Open(original.Flatten())
    path = '/World/TE_J35FreeSplitPlug/Joints/CouplingNutRevolute'
    p = stage.GetPrimAtPath(path); joint = UsdPhysics.Joint(p)
    if p.GetTypeName() != 'PhysicsJoint' or p.GetAttribute('kcg:rotationCoordinate').Get():
        raise ValueError('Expected existing Z-coordinate captive D6 joint')
    limits = UsdPhysics.LimitAPI(p, 'transZ')
    low, high = float(limits.GetLowAttr().Get()), float(limits.GetHighAttr().Get())
    old_drive = UsdPhysics.DriveAPI(p, 'rotZ')
    values = {name: getattr(old_drive, 'Get'+name+'Attr')().Get() for name in
              ('Type', 'Stiffness', 'Damping', 'MaxForce', 'TargetPosition', 'TargetVelocity')}
    old_frames = [joint.GetLocalRot0Attr().Get(), joint.GetLocalRot1Attr().Get()]
    c = Gf.Quatf(float(np.sqrt(.5)), Gf.Vec3f(0., -float(np.sqrt(.5)), 0.))
    new_frames = [q*c for q in old_frames]
    for old, new in zip(old_frames, new_frames):
        before = np.asarray(Gf.Rotation(Gf.Quatd(old)).TransformDir(Gf.Vec3d(0, 0, 1)))
        after = np.asarray(Gf.Rotation(Gf.Quatd(new)).TransformDir(Gf.Vec3d(1, 0, 0)))
        if not np.allclose(before, after, atol=2e-7, rtol=0):
            raise ValueError('Physical shaft direction changed')
    for prop in list(p.GetProperties()):
        if prop.GetName().startswith(('limit:', 'drive:')):
            p.RemoveProperty(prop.GetName())
    tokens = p.GetMetadata('apiSchemas').GetAddedOrExplicitItems()
    p.SetMetadata('apiSchemas', Sdf.TokenListOp.CreateExplicit([
        t for t in tokens if not t.startswith(('PhysicsLimitAPI:', 'PhysicsDriveAPI:'))]))
    for axis in ('transY', 'transZ', 'rotY', 'rotZ'):
        lim = UsdPhysics.LimitAPI.Apply(p, axis); lim.CreateLowAttr(1.); lim.CreateHighAttr(-1.)
    lim = UsdPhysics.LimitAPI.Apply(p, 'transX'); lim.CreateLowAttr(low); lim.CreateHighAttr(high)
    drive = UsdPhysics.DriveAPI.Apply(p, 'rotX')
    for name, value in values.items():
        getattr(drive, 'Create'+name+'Attr')(value)
    joint.GetLocalRot0Attr().Set(new_frames[0]); joint.GetLocalRot1Attr().Set(new_frames[1])
    p.CreateAttribute('kcg:rotationCoordinate', Sdf.ValueTypeNames.Token).Set('rotX')
    p.CreateAttribute('kcg:axialCoordinate', Sdf.ValueTypeNames.Token).Set('transX')
    stage.Flatten().Export(str(output/'connector_model.usdc'))
    record = {'scope': 'CAPTIVE_D6_CONTINUOUS_TWIST_AXIS_CORRECTION', 'source_model': str(source),
              'physical_axial_direction_in_body': [0., 0., 1.], 'new_joint_axial_and_twist_axis': 'X',
              'old_frames_wxyz': [[q.GetReal(), *q.GetImaginary()] for q in old_frames],
              'new_frames_wxyz': [[q.GetReal(), *q.GetImaginary()] for q in new_frames],
              'same_physical_axial_limits_m': [low, high], 'same_drive_parameters': values,
              'same_four_physical_locked_axes_and_two_free_axes': True,
              'body_poses_geometry_mass_inertia_materials_and_socket_unchanged': True,
              'external_attachment_added': False,
              'reference': 'https://nvidia-omniverse.github.io/PhysX/physx/5.1.1/docs/Joints.html#d6-joint',
              'reason': 'Continuous nut rotation used a swing axis with a documented180deg drive singularity; cylindrical joint should use axialX plus twistX',
              'dynamic_validation': 'PENDING'}
    report = json.loads(source.with_name('connector_model_assembly_scene.json').read_text())
    report['captive_twist_axis_correction'] = record
    report['captive_nut_axial_play']['axial_coordinate'] = 'transX'
    report['captive_nut_axial_play']['rotation_coordinate'] = 'rotX'
    (output/'connector_model_assembly_scene.json').write_text(json.dumps(report, indent=2)+'\n')
    (output/'preparation.json').write_text(json.dumps(record, indent=2)+'\n')
    shutil.copy2(source.with_name('install_model.py'), output/'install_model.py')
    update_installer_axis_validation(output/'install_model.py')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-model', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); prepare(a.source_model, a.output)
