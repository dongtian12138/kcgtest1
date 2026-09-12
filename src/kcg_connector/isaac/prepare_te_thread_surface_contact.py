"""Keep continuous nut SDF normals while distributing static socket contacts.

Only existing socket triangles are partitioned. No closed sector caps, new
material volume, robot constraint, mass or pose change is introduced.
"""
from pathlib import Path
import argparse,json,shutil
import numpy as np
from pxr import Sdf,Usd,UsdGeom,UsdPhysics,UsdShade,Vt

NUT='/World/TE_J35FreeSplitPlug/CouplingNut/SourceCadCollision'
SOCKET='/World/TEVisualHandoff/FixedReceptaclePose/OfficialVisual/Geometry'

def prepare(source,output):
    output.mkdir(parents=True,exist_ok=False)
    original=Usd.Stage.Open(str(source.resolve()));stage=Usd.Stage.Open(original.Flatten())
    socket=UsdGeom.Mesh(stage.GetPrimAtPath(SOCKET));vertices=np.asarray(socket.GetPointsAttr().Get(),np.float32)
    counts=np.asarray(socket.GetFaceVertexCountsAttr().Get());faces=np.asarray(socket.GetFaceVertexIndicesAttr().Get(),np.int32).reshape(-1,3)
    if not np.all(counts==3):raise ValueError('Expected source triangles')
    if socket.GetPrim().HasAPI(UsdPhysics.RigidBodyAPI):raise ValueError('Expected original static socket surface')
    all_colliders=[p.GetPath() for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    nut=stage.GetPrimAtPath(NUT);UsdPhysics.CollisionAPI(nut).GetCollisionEnabledAttr().Set(True)
    filtered=UsdPhysics.FilteredPairsAPI.Apply(nut).CreateFilteredPairsRel()
    filtered.SetTargets(list(dict.fromkeys([*filtered.GetTargets(),Sdf.Path(SOCKET)])))
    for i in range(8):
        p=stage.GetPrimAtPath(NUT+f'_SocketSector_{i:02d}')
        if not p:raise ValueError('Expected the current eight-part comparison baseline')
        UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(False)
    centers=vertices[faces].mean(axis=1)
    assignment=np.floor((np.arctan2(centers[:,1],centers[:,0])%(2*np.pi))/(2*np.pi/8)).astype(int)%8
    material,_=UsdShade.MaterialBindingAPI(socket.GetPrim()).ComputeBoundMaterial('physics')
    installed=[];visited=[]
    for i in range(8):
        selected=np.flatnonzero(assignment==i);part=faces[selected];used,inverse=np.unique(part,return_inverse=True);vv=vertices[used];ff=inverse.reshape(-1,3).astype(np.int32)
        path=f'/World/TEVisualHandoff/FixedReceptaclePose/OfficialVisual/NutContactSurface_{i:02d}'
        Sdf.CopySpec(stage.GetRootLayer(),Sdf.Path(SOCKET),stage.GetRootLayer(),Sdf.Path(path))
        g=UsdGeom.Mesh(stage.GetPrimAtPath(path));g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vv));g.CreateFaceVertexCountsAttr([3]*len(ff));g.CreateFaceVertexIndicesAttr(ff.ravel().tolist());g.CreateExtentAttr([vv.min(0).tolist(),vv.max(0).tolist()]);g.CreateVisibilityAttr('invisible')
        g.GetNormalsAttr().Clear()
        p=g.GetPrim();UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr('none')
        UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(True)
        UsdPhysics.FilteredPairsAPI.Apply(p).CreateFilteredPairsRel().SetTargets([x for x in all_colliders if str(x)!=NUT])
        UsdShade.MaterialBindingAPI.Apply(p).Bind(material,UsdShade.Tokens.strongerThanDescendants,'physics')
        # Check exact original float32 vertex coordinates, triangle order and winding.
        if not np.array_equal(vv[ff],vertices[faces[selected]]):raise ValueError('Partition altered an original triangle')
        visited.extend(selected.tolist());installed.append({'path':path,'triangle_count':len(ff),'added_cap_triangles':0})
    if sorted(visited)!=list(range(len(faces))):raise ValueError('Source triangles were lost or repeated')
    before={str(p.GetPath()):{a.GetName():str(a.Get()) for a in p.GetAttributes() if a.GetName().startswith(('physics:mass','physics:centerOfMass','physics:diagonalInertia','physics:principalAxes','xformOp'))} for p in original.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    after={str(p.GetPath()):{a.GetName():str(a.Get()) for a in p.GetAttributes() if a.GetName().startswith(('physics:mass','physics:centerOfMass','physics:diagonalInertia','physics:principalAxes','xformOp'))} for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    if before!=after:raise ValueError('Changed an original actor or its mass/pose')
    stage.Flatten().Export(str(output/'connector_model.usdc'))
    report={'scope':'CONTINUOUS_NUT_SDF_WITH_UNCAPPED_SOCKET_SURFACE_PARTITIONS','source_model':str(source.resolve()),'source_socket_triangle_count':len(faces),'installed_static_patches':installed,'original_triangles_exactly_once':True,'added_cap_triangles':0,'nut_geometry_material_sdf_and_mass_unchanged':True,'original_socket_body_pin_and_stop_contacts_preserved':True,'eight_artificial_nut_sectors_disabled':True,'new_patches_contact_only_original_whole_nut':True,'dynamic_validation':'PENDING'}
    (output/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    scene_path=source.with_name('connector_model_assembly_scene.json')
    if scene_path.exists():
        scene=json.loads(scene_path.read_text());scene['continuous_thread_surface_repair']=report;(output/'connector_model_assembly_scene.json').write_text(json.dumps(scene,indent=2)+'\n')
    installer=source.with_name('install_model.py')
    if installer.exists():shutil.copy2(installer,output/'install_model.py')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.source,a.output)
