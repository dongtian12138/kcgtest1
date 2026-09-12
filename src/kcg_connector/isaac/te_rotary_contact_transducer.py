"""External, frictionless torsion coupling with impulse-based force readout.

Three tabs on the measured nut engage tangential faces on the declared external
spindle. The long faces permit the entire axial stroke. These are removable
apparatus contacts, not connector geometry or an extra connector mass.
"""
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def author(stage, nut_path, rotor_path):
    material = UsdShade.Material.Define(stage, '/World/ExternalTorqueTransducerMaterial')
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(0.); api.CreateDynamicFrictionAttr(0.); api.CreateRestitutionAttr(0.)
    original = [p.GetPath() for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    paths = []
    for index in range(3):
        angle = index*120.
        phi = np.deg2rad(angle)
        rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), angle).GetQuat()
        for label, actor, xyz, scale in [
            ('Tab', nut_path, [.035, 0., -.04], [.004, .002, .004]),
            ('NegativeWall', rotor_path, [.035, -.0015, .05], [.004, .001, .06]),
            ('PositiveWall', rotor_path, [.035, .0015, .05], [.004, .001, .06]),
        ]:
            x, y, z = xyz
            position = [x*np.cos(phi)-y*np.sin(phi), x*np.sin(phi)+y*np.cos(phi), z]
            path = actor+f'/ExternalTorque{label}_{index}'
            cube = UsdGeom.Cube.Define(stage, path); cube.CreateSizeAttr(1.)
            cube.AddTranslateOp().Set(Gf.Vec3d(*position)); cube.AddOrientOp().Set(Gf.Quatf(rotation))
            cube.AddScaleOp().Set(Gf.Vec3f(*scale)); cube.CreateVisibilityAttr('invisible')
            p = cube.GetPrim(); UsdPhysics.CollisionAPI.Apply(p).CreateCollisionEnabledAttr(False)
            p.SetMetadata('apiSchemas', Sdf.TokenListOp.CreateExplicit([*p.GetAppliedSchemas(), 'PhysxCollisionAPI']))
            p.CreateAttribute('physxCollision:contactOffset', Sdf.ValueTypeNames.Float).Set(.00002)
            p.CreateAttribute('physxCollision:restOffset', Sdf.ValueTypeNames.Float).Set(0.)
            UsdPhysics.FilteredPairsAPI.Apply(p).CreateFilteredPairsRel().SetTargets(original)
            UsdShade.MaterialBindingAPI.Apply(p).Bind(material, UsdShade.Tokens.strongerThanDescendants, 'physics')
            resolved, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial('physics')
            if resolved.GetPath() != material.GetPath():
                raise ValueError('An ancestor overrides the torque-transducer material')
            paths.append(path)
    return paths


def set_enabled(stage, paths, enabled):
    for path in paths:
        UsdPhysics.CollisionAPI(stage.GetPrimAtPath(path)).GetCollisionEnabledAttr().Set(enabled)


def read_torque(contact_pairs, shaft_axis_world):
    moment = np.zeros(3)
    count = 0
    for pair in contact_pairs:
        if '/ExternalTorqueTab_' in pair['own_collider']:
            moment += np.asarray(pair['normal_wrench_n_nm'][3:], float)
            count += pair['normal_count']
    return float(moment@np.asarray(shaft_axis_world)), count
