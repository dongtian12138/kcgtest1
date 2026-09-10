#!/usr/bin/env python3
"""Source-contained solid contacts for the five original polarization keys.

The original CAD, clearance, mass, passive joint, friction and full source SDF
remain. These native convex cells occupy existing metal and expose its actual
bearing faces; no world attachment, state-dependent latch or pose correction.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt


def prepare(source_model, output):
    source_model, output = Path(source_model).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Usd.Stage.Open(str(source_model)); stage = Usd.Stage.Open(source.Flatten())
    body = '/World/TE_J35FreeSplitPlug/Body'; core_path = body+'/SocketRigidCoreCollision'
    prim = stage.GetPrimAtPath(core_path); mesh = UsdGeom.Mesh(prim)
    if UsdGeom.Xformable(prim).GetOrderedXformOps():
        raise ValueError('Source core must retain the Body-local metre frame')
    vertices = np.asarray(mesh.GetPointsAttr().Get(), float)
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), int).reshape(-1, 3)
    solid = trimesh.Trimesh(vertices*1000., faces, process=False)
    if not solid.is_volume: raise ValueError('Source core is not a closed solid')
    radius = np.linalg.norm(vertices[:, :2], axis=1)
    cells=[]
    slab=(vertices[:,2]>=-.0076461)&(vertices[:,2]<=-.000761)
    key_faces=faces[np.all(slab[faces],axis=1)&np.any((radius>.0183)[faces],axis=1)]
    patches=trimesh.Trimesh(vertices*1000.,key_faces,process=True).split(only_watertight=False)
    if len(patches)!=5:raise ValueError(f'Expected exactly five source keys, got {len(patches)}')
    patches=sorted(patches,key=lambda p:np.arctan2(p.vertices[:,1].mean(),p.vertices[:,0].mean()))
    for i,patch in enumerate(patches):
        hull=trimesh.convex.convex_hull(patch.vertices)
        if not hull.is_volume or len(hull.vertices)>255:raise ValueError('Source key hull is invalid')
        cells.append(('Key',i,hull))
    samples=[]
    for _,_,hull in cells:
        center=hull.vertices.mean(0)
        surface=np.vstack([hull.vertices,hull.triangles_center,
                           hull.vertices[hull.edges_unique].mean(axis=1)])
        samples.extend(np.vstack([surface, .25*surface+.75*center,.5*surface+.5*center,.75*surface+.25*center]))
    signed=trimesh.proximity.signed_distance(solid,np.asarray(samples))/1000.
    if signed.min() < -2e-8:
        np.savez_compressed(output/'failed_containment.npz',points_mm=samples,signed_distance_m=signed)
        raise ValueError(f'Cells extend outside source metal: {signed.min()} m')
    filters=UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()
    materials=[(r.GetName(),r.GetTargets()) for r in prim.GetRelationships() if r.GetName().startswith('material:binding')]
    paths=[]
    for kind,index,hull in cells:
        path=f'{body}/SourceGuide{kind}_{index:03d}'
        points=hull.vertices/1000.;center=points.mean(0);unit=.001
        m=UsdGeom.Mesh.Define(stage,path);p=m.GetPrim()
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(((points-center)/unit).astype(np.float32)))
        m.CreateFaceVertexCountsAttr([3]*len(hull.faces));m.CreateFaceVertexIndicesAttr(np.asarray(hull.faces).ravel().tolist())
        m.AddTranslateOp().Set(Gf.Vec3d(*center));m.AddScaleOp().Set(Gf.Vec3f(unit))
        m.CreateSubdivisionSchemeAttr('none');m.CreateVisibilityAttr('invisible')
        UsdPhysics.CollisionAPI.Apply(p).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr('convexHull')
        p.SetMetadata('apiSchemas',Sdf.TokenListOp.CreateExplicit([*p.GetAppliedSchemas(),'PhysxCollisionAPI','PhysxConvexHullCollisionAPI']))
        p.CreateAttribute('physxConvexHullCollision:hullVertexLimit',Sdf.ValueTypeNames.Int).Set(255)
        p.CreateAttribute('physxConvexHullCollision:minThickness',Sdf.ValueTypeNames.Float).Set(0.)
        for name in ('physxCollision:contactOffset','physxCollision:restOffset'):
            attr=prim.GetAttribute(name);p.CreateAttribute(name,attr.GetTypeName()).Set(attr.Get())
        UsdPhysics.FilteredPairsAPI.Apply(p).CreateFilteredPairsRel().SetTargets(filters)
        UsdShade.MaterialBindingAPI.Apply(p)
        for name,targets in materials:p.CreateRelationship(name).SetTargets(targets)
        paths.append(path)
    stage.Flatten().Export(str(output/'connector_model.usdc'))
    # No property on any original prim may change as a side effect.
    check=Usd.Stage.Open(str(output/'connector_model.usdc'))
    for p in source.Traverse():
        q=check.GetPrimAtPath(p.GetPath())
        for prop in p.GetProperties():
            equal=(str(prop.Get())==str(q.GetAttribute(prop.GetName()).Get()) if isinstance(prop,Usd.Attribute)
                   else prop.GetTargets()==q.GetRelationship(prop.GetName()).GetTargets())
            if not equal:raise ValueError('An original property changed: '+str(prop.GetPath()))
    audit={'scope':'SOURCE_CONTAINED_SOLID_GUIDE_CONTACT_REPRESENTATION',
           'source_model':str(source_model),'source_core':core_path,'installed_paths':paths,
           'shell_sector_count':0,'source_key_count':5,'mesh_coordinate_unit_m':.001,
           'containment_sample_count':len(samples),'minimum_sample_signed_inside_distance_m':float(signed.min()),
           'containment_scope':'Hull vertices, edges, face centers and interior rays checked against source solid',
           'original_properties_unchanged_check_passed':True,'source_material_filters_mass_inertia_joint_unchanged':True,
           'world_joint_or_pose_servo_added':False,'dynamic_validation':'PENDING'}
    report=json.loads(source_model.with_name('connector_model_assembly_scene.json').read_text());report['source_shell_guide_contacts']=audit
    (output/'connector_model_assembly_scene.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'preparation.json').write_text(json.dumps(audit,indent=2)+'\n')
    shutil.copy2(source_model.with_name('install_model.py'),output/'install_model.py')
    print(json.dumps({k:v for k,v in audit.items() if k!='installed_paths'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-model',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();prepare(args.source_model,args.output)
