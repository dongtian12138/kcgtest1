#!/usr/bin/env python3
"""Install native elastic tooth/pawl contact from the visible FreeCAD candidate.

The topology follows the cited Deutsch patent; dimensions are explicitly
representative. Elastic contact replaces unresolved leaf bending without adding
mass, a world attachment, an angular lock or state-dependent actuation.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

BODY = '/World/TE_J35FreeSplitPlug/Body'
NUT = '/World/TE_J35FreeSplitPlug/CouplingNut'
JOINT = '/World/TE_J35FreeSplitPlug/Joints/CouplingNutRevolute'


def prepare(source_path, cad_path, output, stiffness):
    source_path, cad_path, output = map(lambda p: Path(p).resolve(), (source_path, cad_path, output))
    output.mkdir(parents=True, exist_ok=False)
    source = Usd.Stage.Open(str(source_path)); stage = Usd.Stage.Open(source.Flatten())
    cad = json.loads(cad_path.read_text())
    joint = stage.GetPrimAtPath(JOINT)
    if joint.GetAttribute('kcg:rotationCoordinate').Get() != 'rotX':
        raise ValueError('Expected the verified continuous-twist bearing')
    old_colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    external = [p.GetPath() for p in old_colliders if not str(p.GetPath()).startswith((BODY+'/', NUT+'/'))]
    # Permit only the new tooth/pawl interfaces between the existing actors.
    for p in old_colliders:
        path = str(p.GetPath())
        opposite = NUT if path.startswith(BODY+'/') else BODY if path.startswith(NUT+'/') else None
        if opposite:
            api = UsdPhysics.FilteredPairsAPI.Apply(p)
            api.CreateFilteredPairsRel().SetTargets(list(dict.fromkeys([*api.GetFilteredPairsRel().GetTargets(), Sdf.Path(opposite)])))
    UsdPhysics.Joint(joint).CreateCollisionEnabledAttr(True)
    joint.CreateAttribute('kcg:representativeRatchetContact', Sdf.ValueTypeNames.Bool).Set(True)
    # Native tooth/pawl friction now supplies the passive rotational resistance.
    # Retire the old guessed constant brake rather than count it twice.
    old_brake = UsdPhysics.DriveAPI(joint, 'rotX')
    old_brake.GetMaxForceAttr().Set(0.); old_brake.GetDampingAttr().Set(0.)
    joint.GetAttribute('kcg:passiveResistanceNm').Set(0.)
    joint.CreateAttribute('kcg:passiveResistanceLaw', Sdf.ValueTypeNames.String).Set('native_ratchet_and_thrust_contact')
    material_path = '/World/ConnectorFrozenModelMaterials/RepresentativeRatchetElasticContact'
    material = UsdShade.Material.Define(stage, material_path)
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(.45); physics.CreateDynamicFrictionAttr(.45); physics.CreateRestitutionAttr(0.)
    material.GetPrim().SetMetadata('apiSchemas', Sdf.TokenListOp.CreateExplicit([
        *material.GetPrim().GetAppliedSchemas(), 'PhysxMaterialAPI']))
    for key, value, kind in [('compliantContactStiffness', stiffness, Sdf.ValueTypeNames.Float),
                             ('compliantContactDamping', 1., Sdf.ValueTypeNames.Float),
                             ('compliantContactAccelerationSpring', False, Sdf.ValueTypeNames.Bool),
                             ('frictionCombineMode', 'max', Sdf.ValueTypeNames.Token)]:
        material.GetPrim().CreateAttribute('physxMaterial:'+key, kind).Set(value)
    metal, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(BODY+'/SocketRigidCoreCollision')).ComputeBoundMaterial('physics')
    installed = []

    def contact(prim, mat, convex=False):
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        tokens = [*prim.GetAppliedSchemas(), 'PhysxCollisionAPI']
        if convex:
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
            tokens = [*prim.GetAppliedSchemas(), 'PhysxCollisionAPI', 'PhysxConvexHullCollisionAPI']
        prim.SetMetadata('apiSchemas', Sdf.TokenListOp.CreateExplicit(list(dict.fromkeys(tokens))))
        prim.CreateAttribute('physxCollision:contactOffset', Sdf.ValueTypeNames.Float).Set(.000002)
        prim.CreateAttribute('physxCollision:restOffset', Sdf.ValueTypeNames.Float).Set(0.)
        if convex:
            prim.CreateAttribute('physxConvexHullCollision:minThickness', Sdf.ValueTypeNames.Float).Set(0.)
            prim.CreateAttribute('physxConvexHullCollision:hullVertexLimit', Sdf.ValueTypeNames.Int).Set(64)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants, 'physics')
        UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().SetTargets(external)
        prim.CreateAttribute('kcg:internalJointContactOnly',Sdf.ValueTypeNames.Bool).Set(True)
        UsdGeom.Imageable(prim).CreateVisibilityAttr('invisible')
        installed.append(str(prim.GetPath()))

    for tooth in cad['teeth']:
        path = f"{BODY}/RatchetTooth_{tooth['ring']}_{tooth['index']:02d}"
        hull = trimesh.convex.convex_hull(np.asarray(tooth['vertices_mm'], float))
        if not hull.is_volume or len(hull.vertices) > 64:
            raise ValueError('FreeCAD tooth did not produce a valid small convex')
        center = hull.vertices.mean(0)
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy((hull.vertices-center).astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr([3]*len(hull.faces)); mesh.CreateFaceVertexIndicesAttr(hull.faces.ravel().tolist())
        mesh.CreateSubdivisionSchemeAttr('none'); mesh.AddTranslateOp().Set(Gf.Vec3d(*(center/1000.)))
        mesh.AddScaleOp().Set(Gf.Vec3f(.001)); contact(mesh.GetPrim(), metal, convex=True)
    for pawl in cad['pawls']:
        path = f"{NUT}/RatchetPawl_{pawl['ring']}_{pawl['index']}"
        sphere = UsdGeom.Sphere.Define(stage, path); sphere.CreateRadiusAttr(pawl['radius_mm']/1000.)
        sphere.AddTranslateOp().Set(Gf.Vec3d(*(np.asarray(pawl['center_mm'])/1000.)))
        contact(sphere.GetPrim(), material)
    for actor in (BODY, NUT):
        for key in ('physics:mass', 'physics:centerOfMass', 'physics:diagonalInertia', 'physics:principalAxes'):
            if str(source.GetPrimAtPath(actor).GetAttribute(key).Get()) != str(stage.GetPrimAtPath(actor).GetAttribute(key).Get()):
                raise ValueError('Original actor mass properties changed')
    stage.Flatten().Export(str(output/'connector_model.usdc'))
    record = {'scope': 'NATIVE_ELASTIC_RATCHET_CANDIDATE_REQUIRES_CALIBRATION_AND_ASSEMBLY_CHECK',
              'source_model': str(source_path), 'freecad_geometry': str(cad_path),
              'geometry_parameters': cad['parameters'], 'per_contact_stiffness_n_m': stiffness,
              'per_contact_damping_ns_m': 1., 'contact_friction': .45, 'retired_constant_brake_cap_nm': .02,
              'native_tooth_count': len(cad['teeth']), 'native_pawl_contacts': len(cad['pawls']),
              'passive_mechanism': 'Existing Body/Nut actors and native compliant contacts; no control callback',
              'original_masses_inertia_external_geometry_threads_keys_and_axial_limits_unchanged': True,
              'additional_dynamic_bodies_or_world_joints': False,
              'manufacturer_exact_dimensions_or_torque_curve': False,
              'installed_contact_paths': installed}
    (output/'preparation.json').write_text(json.dumps(record, indent=2)+'\n')
    report = json.loads(source_path.with_name('connector_model_assembly_scene.json').read_text())
    report['ratchet_contact_candidate'] = record
    report['candidate_validation'] = 'PENDING: prior assembly results refer to the model without these contacts'
    (output/'connector_model_assembly_scene.json').write_text(json.dumps(report, indent=2)+'\n')
    shutil.copy2(source_path.with_name('install_model.py'), output/'install_model.py')
    print(json.dumps({k: v for k, v in record.items() if k != 'installed_contact_paths'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-model', type=Path, required=True)
    parser.add_argument('--cad-geometry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stiffness-n-m', type=float, default=10000.)
    args = parser.parse_args()
    if not 0 < args.stiffness_n_m < 1e6: parser.error('Finite elastic contact stiffness required')
    prepare(args.source_model, args.cad_geometry, args.output, args.stiffness_n_m)
