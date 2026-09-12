#!/usr/bin/env python3
"""Bounded pin-entry contact compliance candidate; never a rigid-body lock.

Retains every source vertex, collision filter, friction coefficient, body mass
and joint. Only the socket-specific pin contact law changes. A native contact
spring is a local surrogate for elastic contact/beam displacement, not a beam
solver or an identified TE/J599 material model. Keep it out of the baseline
until the declared short physical comparison has been reviewed.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def prepare(source_model, output):
    source_model, output = Path(source_model).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Usd.Stage.Open(str(source_model))
    stage = Usd.Stage.Open(source.Flatten())
    body = '/World/TE_J35FreeSplitPlug/Body'
    official = [p for p in Usd.PrimRange(stage.GetPrimAtPath(body))
                if p.GetName().startswith(('OfficialPinShaft_', 'OfficialPinNose_'))]
    pin_prims = official or [p for p in Usd.PrimRange(stage.GetPrimAtPath(body)) if p.GetName().startswith('SourcePinSdf_')]
    pin_ids = ([int(p.GetAttribute('kcg:sourcePinIndex').Get()) for p in official if p.GetName().startswith('OfficialPinNose_')]
               if official else [int(i) for p in pin_prims for i in p.GetAttribute('kcg:sourcePinIndices').Get()])
    if sorted(pin_ids) != list(range(128)):
        raise ValueError('All original 128 pins must be present exactly once')
    repo = Path(__file__).resolve().parents[3]
    geometry_path = repo/'artifacts/kcg_connector/isaac/te_full_assembly_20260905/source_socket_blind_bore_faces_v1.json'
    geometry = json.loads(geometry_path.read_text())
    radius = geometry['plug_pin_cylinder_radius_m']
    lower, upper = geometry['plug_pin_cylinder_z_m']
    length = upper-lower
    # Representative copper-alloy modulus, NOT an assertion about this part's
    # alloy. 120 GPa is an explicitly assumed reference scale, not a fitted
    # parameter. 3EI/L^3 supplies a pin-tip bending scale from the source CAD.
    modulus = 120e9
    second_moment = np.pi*radius**4/4
    stiffness = 3*modulus*second_moment/length**3
    material, _ = UsdShade.MaterialBindingAPI(pin_prims[0]).ComputeBoundMaterial('physics')
    destination = '/World/ConnectorFrozenModelMaterials/RepresentativePinEntryCompliance'
    if not Sdf.CopySpec(stage.GetRootLayer(), material.GetPath(), stage.GetRootLayer(), Sdf.Path(destination)):
        raise RuntimeError('Could not copy source friction material')
    new_material = UsdShade.Material.Get(stage, destination)
    mp = new_material.GetPrim()
    mp.CreateAttribute('physxMaterial:compliantContactStiffness', Sdf.ValueTypeNames.Float).Set(float(stiffness))
    mp.CreateAttribute('physxMaterial:compliantContactDamping', Sdf.ValueTypeNames.Float).Set(0.)
    mp.CreateAttribute('physxMaterial:compliantContactAccelerationSpring', Sdf.ValueTypeNames.Bool).Set(False)
    # PhysX stores compliant stiffness as NEGATIVE restitution. eMAX therefore
    # chooses the softer of two compliant materials, preserving the existing
    # 2184.98 N/m socket clips instead of averaging them with the pin stiffness.
    # A hard/soft pair always uses the soft side regardless of this setting.
    # See PxsMaterialCombiner.h, PxsCombineMaterials, exactlyOneCompliant branch.
    mp.CreateAttribute('physxMaterial:restitutionCombineMode', Sdf.ValueTypeNames.Token).Set('max')
    for p in pin_prims:
        old_material, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial('physics')
        if old_material.GetPath() != material.GetPath(): raise ValueError('Pin materials differ')
        UsdShade.MaterialBindingAPI.Apply(p).Bind(new_material, UsdShade.Tokens.strongerThanDescendants, 'physics')
    model_path = output/'connector_model.usdc'
    stage.Flatten().Export(str(model_path))
    # Compare the entire stage property set: the only existing properties
    # allowed to differ are the pin-group material-binding relationships.
    after = Usd.Stage.Open(str(model_path))
    changed = []
    allowed = {str(p.GetPath())+'.material:binding:physics' for p in pin_prims}
    for p in source.Traverse():
        q = after.GetPrimAtPath(p.GetPath())
        if not q: raise ValueError('Source prim removed')
        for prop in p.GetProperties():
            name = str(prop.GetPath())
            if isinstance(prop, Usd.Attribute):
                equal = str(prop.Get()) == str(q.GetAttribute(prop.GetName()).Get())
            else:
                equal = prop.GetTargets() == q.GetRelationship(prop.GetName()).GetTargets()
            if not equal:
                changed.append(name)
                if name not in allowed: raise ValueError('Unexpected source change: '+name)
    if set(changed) != allowed: raise ValueError('Expected all pin material bindings to change')
    report_path = source_model.with_name('connector_model_assembly_scene.json')
    report = json.loads(report_path.read_text())
    record = {
        'scope': 'PIN_ENTRY_COMPLIANCE_CANDIDATE_NOT_HARDWARE_CALIBRATION',
        'source_model': str(source_model), 'source_geometry': str(geometry_path),
        'manufacturer_material_reference': 'https://www.te.com/en/products/connectors/circular-connectors/intersection/design-mil-dtl-38999-connectors.html',
        'reference_supports': 'Copper-alloy contacts and thermoplastic contact retention; it supplies no modulus or stiffness for this part',
        'assumed_reference_modulus_pa': modulus, 'pin_radius_m': radius, 'free_shaft_length_m': length,
        'pin_tip_bending_scale_n_m': float(stiffness), 'formula': '3 E (pi r^4 / 4) / L^3',
        'native_per_contact_stiffness_n_m': float(stiffness), 'native_damping_ns_m': 0.,
        'native_restitution_combine_mode': 'max',
        'soft_soft_rule': 'max of negative stiffness: retains the softer existing clip contact law',
        'hard_soft_rule': 'PhysX exactlyOneCompliant branch selects the compliant pin law',
        'mixing_rule_reference': 'artifacts/kcg_connector/model_delivery_20260908/src/contact_physx_PxsMaterialCombiner.h:67',
        'approximation_limits': [
            'Local contact penetration stands in for displacement; no visible shaft bending or coupled beam solution',
            'Each contact point receives the single-pin stiffness scale; several points act in parallel, so this is not the total calibrated pin stiffness',
            'Isotropic native contact spring also acts on tip/end-face normals; deeper insertion/axial bottoming is not validated by an entry test',
            'Exact alloy, retention compliance and elastic limit are unknown; do not infer hardware allowable loads'],
        'pin_count': 128, 'collider_count': len(pin_prims), 'changed_properties': changed,
        'official_rounded_pin_geometry_retained': bool(official),
        'material_path': destination, 'source_mesh_mass_inertia_joints_filters_friction_unchanged': True,
        'external_constraints_added': False, 'post_start_pose_writes': False,
        'physical_validation': 'PENDING_SHORT_COMPARISON', 'source_assets_overwritten': False}
    report['pin_entry_compliance_candidate'] = record
    (output/'connector_model_assembly_scene.json').write_text(json.dumps(report, indent=2)+'\n')
    (output/'preparation.json').write_text(json.dumps(record, indent=2)+'\n')
    shutil.copy2(source_model.with_name('install_model.py'), output/'install_model.py')
    print(json.dumps({k: record[k] for k in ('scope', 'pin_count', 'collider_count', 'pin_tip_bending_scale_n_m', 'source_mesh_mass_inertia_joints_filters_friction_unchanged')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source_model, args.output)
