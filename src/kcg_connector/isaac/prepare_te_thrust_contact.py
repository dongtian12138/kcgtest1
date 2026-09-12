#!/usr/bin/env python3
"""Candidate native frictional shoulders for the captive Body/Nut bearing.

The supplier surface CAD does not resolve the retaining collar. These internal
annular contact cells are an explicitly representative bearing, using the two
source partition radii, the user's estimated travel and the existing friction.
They do not change the five keys, socket, threads, mass, or world attachments.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

BODY = '/World/TE_J35FreeSplitPlug/Body'
NUT = '/World/TE_J35FreeSplitPlug/CouplingNut'
JOINT = '/World/TE_J35FreeSplitPlug/Joints/CouplingNutRevolute'


def prepare(source_model, output):
    source_model, output = Path(source_model).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Usd.Stage.Open(str(source_model))
    stage = Usd.Stage.Open(source.Flatten())
    joint = stage.GetPrimAtPath(JOINT)
    travel = float(joint.GetAttribute('kcg:captiveNutAxialPlayM').Get())
    if joint.GetTypeName() != 'PhysicsJoint' or abs(travel-.001) > 1e-9:
        raise ValueError('Expected the existing four-axis bearing with 1 mm total travel')
    manifest = Path(__file__).resolve().parents[3] / 'artifacts/kcg_connector/isaac/te_j35_free_split_tabletop_v1/MANIFEST.json'
    caps = json.loads(manifest.read_text())['parts']['Body']['hidden_interface_caps']
    radii = sorted(c['radial_extent_m'] for c in caps)
    # The rear retaining interface keeps the surrogate shoulders behind the
    # receptacle, throughout the declared nut travel. The front split cap is
    # not an appropriate place for an internal retaining shoulder.
    datum = min(c['plane_z_m'] for c in caps)
    thickness, backup_margin = .0002, .0001
    material, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(BODY+'/SocketRigidCoreCollision')).ComputeBoundMaterial('physics')
    mu = UsdPhysics.MaterialAPI(material.GetPrim())
    if abs(mu.GetStaticFrictionAttr().Get()-.45) > 1e-6:
        raise ValueError('Expected unchanged existing connector friction')
    original_colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    external = [p.GetPath() for p in original_colliders
                if not str(p.GetPath()).startswith((BODY+'/', NUT+'/'))]
    # Enabling internal shoulder contacts must not re-enable collision between
    # the old, artificially capped source Body/Nut surfaces.
    changed_filters = []
    for p in original_colliders:
        path = str(p.GetPath())
        other = NUT if path.startswith(BODY+'/') else BODY if path.startswith(NUT+'/') else None
        if other and not p.GetName().startswith(('RatchetTooth_', 'RatchetPawl_')):
            api = UsdPhysics.FilteredPairsAPI.Apply(p)
            targets = api.GetFilteredPairsRel().GetTargets()
            api.CreateFilteredPairsRel().SetTargets(list(dict.fromkeys([*targets, Sdf.Path(other)])))
            changed_filters.append(path)
    installed = []
    rings = [(BODY, 'Collar', datum-thickness/2, datum+thickness/2),
             (NUT, 'RearShoulder', datum-thickness/2-travel/2-thickness, datum-thickness/2-travel/2),
             (NUT, 'FrontShoulder', datum+thickness/2+travel/2, datum+thickness/2+travel/2+thickness)]
    for actor, label, z0, z1 in rings:
        for index in range(16):
            if actor == BODY:
                # Fixed rounded quadrature pads sample the same annular face.
                # They have no rolling DOF. This avoids tangent-facing internal
                # seam walls from two overlapping convex annulus decompositions.
                a = (index+.5)*2*np.pi/16
                center = [float(np.mean(radii))*np.cos(a), float(np.mean(radii))*np.sin(a), datum]
                path = f'{actor}/RepresentativeThrust_{label}_{index:02d}'
                sphere = UsdGeom.Sphere.Define(stage, path); p = sphere.GetPrim()
                sphere.CreateRadiusAttr(thickness/2); sphere.AddTranslateOp().Set(Gf.Vec3d(*center))
                sphere.CreateVisibilityAttr('invisible')
                UsdPhysics.CollisionAPI.Apply(p).CreateCollisionEnabledAttr(True)
                p.SetMetadata('apiSchemas', Sdf.TokenListOp.CreateExplicit([*p.GetAppliedSchemas(), 'PhysxCollisionAPI']))
                p.CreateAttribute('physxCollision:contactOffset', Sdf.ValueTypeNames.Float).Set(.0001)
                p.CreateAttribute('physxCollision:restOffset', Sdf.ValueTypeNames.Float).Set(0.)
                UsdPhysics.FilteredPairsAPI.Apply(p).CreateFilteredPairsRel().SetTargets(external)
                p.CreateAttribute('kcg:internalJointContactOnly',Sdf.ValueTypeNames.Bool).Set(True)
                UsdShade.MaterialBindingAPI.Apply(p).Bind(material, UsdShade.Tokens.strongerThanDescendants, 'physics')
                installed.append(path)
                continue
            if index:
                continue
            # A single flat proxy per shoulder is evaluated only against the
            # fixed annular contact nodes. Every node stays within the real
            # annulus, so unused proxy area contributes no contact or force.
            path = f'{actor}/RepresentativeThrust_{label}_Plane'
            cube = UsdGeom.Cube.Define(stage, path); p = cube.GetPrim()
            cube.CreateSizeAttr(1.)
            cube.AddTranslateOp().Set(Gf.Vec3d(0., 0., (z0+z1)/2))
            cube.AddScaleOp().Set(Gf.Vec3f(2*radii[-1], 2*radii[-1], z1-z0))
            cube.CreateVisibilityAttr('invisible')
            UsdPhysics.CollisionAPI.Apply(p).CreateCollisionEnabledAttr(True)
            p.SetMetadata('apiSchemas', Sdf.TokenListOp.CreateExplicit([*p.GetAppliedSchemas(), 'PhysxCollisionAPI']))
            p.CreateAttribute('physxCollision:contactOffset', Sdf.ValueTypeNames.Float).Set(.0001)
            p.CreateAttribute('physxCollision:restOffset', Sdf.ValueTypeNames.Float).Set(0.)
            UsdPhysics.FilteredPairsAPI.Apply(p).CreateFilteredPairsRel().SetTargets(external)
            p.CreateAttribute('kcg:internalJointContactOnly',Sdf.ValueTypeNames.Bool).Set(True)
            UsdShade.MaterialBindingAPI.Apply(p).Bind(material, UsdShade.Tokens.strongerThanDescendants, 'physics')
            installed.append(path)
    # Contact carries the working axial load. A separate100micrometre backup
    # limit remains bounded; reaching it invalidates the contact-load test.
    UsdPhysics.Joint(joint).CreateCollisionEnabledAttr(True)
    axial_coordinate = joint.GetAttribute('kcg:axialCoordinate').Get() or 'transZ'
    axial = UsdPhysics.LimitAPI(joint, axial_coordinate)
    axial.CreateLowAttr(-travel/2-backup_margin)
    axial.CreateHighAttr(travel/2+backup_margin)
    joint.CreateAttribute('kcg:representativeThrustContact', Sdf.ValueTypeNames.Bool).Set(True)
    joint.CreateAttribute('kcg:thrustBackupMarginM', Sdf.ValueTypeNames.Float).Set(backup_margin)
    stage.Flatten().Export(str(output/'connector_model.usdc'))
    check = Usd.Stage.Open(str(output/'connector_model.usdc'))
    for p in source.Traverse():
        if str(p.GetPath()) == JOINT:
            continue
        q = check.GetPrimAtPath(p.GetPath())
        for prop in p.GetProperties():
            if str(p.GetPath()) in changed_filters and prop.GetName() == 'physics:filteredPairs':
                continue
            equal = (str(prop.Get()) == str(q.GetAttribute(prop.GetName()).Get()) if isinstance(prop, Usd.Attribute)
                     else prop.GetTargets() == q.GetRelationship(prop.GetName()).GetTargets())
            if not equal:
                raise ValueError('Unexpected original property change: '+str(prop.GetPath()))
    record = {'scope': 'REPRESENTATIVE_INTERNAL_THRUST_CONTACT_CANDIDATE',
              'source_model': str(source_model), 'source_interface_manifest': str(manifest),
              'annulus_radii_m': radii, 'representative_collar_center_z_m': datum,
              'representative_plate_thickness_m': thickness, 'travel_m': travel,
              'axial_coordinate': axial_coordinate, 'retaining_interface': 'rear source split interface',
              'working_axial_limits_m': [-travel/2, travel/2], 'backup_margin_m': backup_margin,
              'backup_margin_basis': 'Outside the measured12um loaded contact-position error and the46um socket SDF cell; physical shoulder positions and1mm freeplay are unchanged',
              'static_and_dynamic_friction': .45,
              'fixed_base_resistance_nm': float(joint.GetAttribute('kcg:passiveResistanceNm').Get()),
              'body_annular_face_discretization': '16 fixed spherical quadrature nodes; not ball bearings or additional moving bodies',
              'nut_shoulder_representation': 'Two flat single-piece contact proxies sampled strictly inside the actual annular region; avoids artificial decomposition seams',
              'minimum_node_edge_margin_m': float(min(np.mean(radii)-radii[0],radii[-1]-np.mean(radii))-thickness/2),
              'contact_detection_offset_per_shape_m': .0001,
              'contact_detection_basis': 'Observed gap take-up step175.6um at960Hz; combined200um detection margin anticipates the contact. Rest offset stays0, preserving1mm working play.',
              'pressure_from': 'NATIVE_BODY_NUT_CONTACT_CONSTRAINTS_NO_EXTERNAL_FORCE_OR_STATE_SERVO',
              'hardware_internal_geometry_and_friction_calibrated': False,
              'source_geometry_mass_inertia_materials_unchanged': True,
              'source_body_nut_contact_still_filtered': True, 'installed_paths': installed,
              'validation': 'PENDING_SAME_PROFILE_SHORT_COMPARISON'}
    report = json.loads(source_model.with_name('connector_model_assembly_scene.json').read_text())
    report['representative_thrust_contact'] = record
    (output/'connector_model_assembly_scene.json').write_text(json.dumps(report, indent=2)+'\n')
    (output/'preparation.json').write_text(json.dumps(record, indent=2)+'\n')
    # This candidate-only adapter recognizes the explicitly marked backed-up
    # native stop. Production adoption requires dynamic validation first.
    adapter = source_model.with_name('install_model.py').read_text()
    needle = "        axial=UsdPhysics.LimitAPI(prim,axial_coordinate)\n        return (abs(float(axial.GetLowAttr().Get())+float(travel)/2)<1e-9 and\n                abs(float(axial.GetHighAttr().Get())-float(travel)/2)<1e-9)"
    replacement = """        margin=0.
        if prim.GetAttribute('kcg:representativeThrustContact').Get():
            margin=float(prim.GetAttribute('kcg:thrustBackupMarginM').Get())
            if abs(margin-.0001)>1e-10 or not UsdPhysics.Joint(prim).GetCollisionEnabledAttr().Get():return False
        axial=UsdPhysics.LimitAPI(prim,axial_coordinate)
        return (abs(float(axial.GetLowAttr().Get())+float(travel)/2+margin)<1e-9 and
                abs(float(axial.GetHighAttr().Get())-float(travel)/2-margin)<1e-9)"""
    if needle not in adapter:
        raise ValueError('Candidate adapter source changed; review required')
    (output/'install_model.py').write_text(adapter.replace(needle, replacement))
    print(json.dumps({k: v for k, v in record.items() if k != 'installed_paths'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source_model, args.output)
