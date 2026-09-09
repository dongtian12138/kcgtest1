"""Source-preserving convex pieces for the stepped original contact pins.

Each pin has a 140-vertex lower ring and a 70-vertex upper ring. Splitting the
lower cylinder into two halves avoids the PhysX convex vertex limit without
simplifying or enlarging the source profile. The remaining core is capped at
the original pin-root plane, inside the original union. Mass stays on Body.
"""
import json
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree,ConvexHull
from pxr import UsdGeom,UsdPhysics,Sdf,Vt,Gf


def source_pin_pieces(vertices,faces,centres):
    vertices=np.asarray(vertices,float);faces=np.asarray(faces,int);centres=np.asarray(centres,float)
    distance,owner=cKDTree(centres).query(vertices[:,:2])
    pin_vertex=(distance<.000374)&(vertices[:,2]>=-.015001)&(vertices[:,2]<=-.00946149)
    removed=np.zeros(len(faces),bool);caps=[];pieces=[];audits=[];solids=[]
    for index,centre in enumerate(centres):
        mask=np.all(pin_vertex[faces]&(owner[faces]==index),axis=1)
        ids=np.unique(faces[mask]);points=vertices[ids]
        unique,reverse=np.unique(points,axis=0,return_index=True);ids=ids[reverse];points=unique
        levels=np.unique(points[:,2])
        if len(levels)!=3:raise ValueError('source stepped pin must have three axial levels')
        low,mid,high=levels
        bottom_ids=ids[points[:,2]==low];bottom=vertices[bottom_ids]
        bottom_ids=bottom_ids[np.argsort(np.arctan2(bottom[:,1]-centre[1],bottom[:,0]-centre[0]))]
        bottom=vertices[bottom_ids]
        middle=points[points[:,2]==mid];top=points[points[:,2]==high]
        small_radius=np.linalg.norm(top[:,:2]-centre,axis=1).mean()
        large_middle=middle[np.linalg.norm(middle[:,:2]-centre,axis=1)>small_radius+.0000004]
        small_middle=middle[np.linalg.norm(middle[:,:2]-centre,axis=1)<small_radius+.0000004]
        if (len(bottom),len(large_middle),len(small_middle),len(top))!=(140,140,70,70):
            raise ValueError('source pin ring identity differs; do not approximate it silently')
        dd,jj=cKDTree(large_middle[:,:2]).query(bottom[:,:2])
        if dd.max()>2e-9:raise ValueError('source lower pin rings do not share the profile')
        large_middle=large_middle[jj]
        root_cap=np.array([[bottom_ids[0],bottom_ids[j],bottom_ids[j+1]] for j in range(1,len(bottom_ids)-1)])
        caps.extend(root_cap)
        original=trimesh.Trimesh(vertices,np.vstack((faces[mask],root_cap[:,::-1])),process=False)
        original.remove_unreferenced_vertices();original.merge_vertices()
        if not original.is_watertight or not original.is_winding_consistent:
            raise ValueError('source pin plus root closure is not a closed consistent solid')
        solids.append(dict(pin=index,vertices=np.asarray(original.vertices).copy(),faces=np.asarray(original.faces).copy()))
        volume=0.;new=[]
        halves=[np.arange(71),np.r_[np.arange(70,140),0]]
        clouds=[np.vstack((bottom[h],large_middle[h])) for h in halves]+[np.vstack((small_middle,top))]
        for part,cloud in enumerate(clouds):
            hull=ConvexHull(cloud);tri=hull.simplices.copy()
            normals=np.cross(cloud[tri[:,1]]-cloud[tri[:,0]],cloud[tri[:,2]]-cloud[tri[:,0]])
            flip=np.einsum('ij,ij->i',normals,hull.equations[:,:3])<0;tri[flip]=tri[flip,::-1]
            solid=trimesh.Trimesh(cloud,tri,process=False);solid.remove_unreferenced_vertices()
            if len(solid.vertices)>255 or not solid.is_watertight or not solid.is_winding_consistent:
                raise ValueError('source pin convex piece violates native hull constraints')
            volume+=abs(solid.volume)
            new.append(dict(pin=index,part=part,vertices=np.asarray(solid.vertices),faces=np.asarray(solid.faces)))
        error=abs(volume-abs(original.volume))/abs(original.volume)
        if error>2e-6:raise ValueError('convex pieces changed source pin volume: '+str(error))
        pieces.extend(new);removed|=mask
        audits.append(dict(pin=index,source_faces=int(mask.sum()),source_closed_volume_m3=abs(original.volume),
            pieces_volume_m3=volume,relative_volume_difference=error,
            axial_levels_m=levels.tolist(),piece_vertex_counts=[len(n['vertices']) for n in new]))
    remaining_faces=np.vstack((faces[~removed],np.asarray(caps,int)))
    remaining=trimesh.Trimesh(vertices,remaining_faces,process=False);remaining.remove_unreferenced_vertices();remaining.merge_vertices()
    if not remaining.is_watertight or not remaining.is_winding_consistent:
        raise ValueError('capped core must remain a closed source solid')
    return remaining_faces,pieces,audits,solids


def install_source_pin_convex_contacts(stage,*,body_path,core_path,source_pin_centres):
    """Call before physics only; return geometry/authoring audit for this candidate."""
    core=UsdGeom.Mesh.Get(stage,core_path);prim=core.GetPrim()
    if not core:raise ValueError('missing source rigid core')
    if len(UsdGeom.Xformable(core).GetOrderedXformOps()):
        raise ValueError('source pin coordinates require identity core-local transform')
    v=np.asarray(core.GetPointsAttr().Get(),float);f=np.asarray(core.GetFaceVertexIndicesAttr().Get(),int).reshape(-1,3)
    centres=np.asarray(json.load(Path(source_pin_centres).open())['source_plug_pin_centers_m'])
    remaining,pieces,audits,_=source_pin_pieces(v,f,centres)
    if len(audits)!=128:raise ValueError('the connector must retain all 128 original pins')
    filters=UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets() if prim.HasAPI(UsdPhysics.FilteredPairsAPI) else []
    material_relations=[(rel.GetName(),rel.GetTargets()) for rel in prim.GetRelationships() if rel.GetName().startswith('material:binding')]
    paths=[]
    for piece in pieces:
        path=f'{body_path}/SourcePinConvex_{piece["pin"]:03d}_{piece["part"]}'
        if stage.GetPrimAtPath(path):raise ValueError('candidate pin path already exists')
        mesh=UsdGeom.Mesh.Define(stage,path);x=mesh.GetPrim()
        centre=piece['vertices'].mean(axis=0)
        # Convex cooking has a raw-coordinate plane tolerance. Normalize only
        # mesh coordinates; the inverse local scale preserves SI geometry.
        coordinate_unit_m=1e-6
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(((piece['vertices']-centre)/coordinate_unit_m).astype(np.float32)))
        mesh.AddTranslateOp().Set(Gf.Vec3d(*centre))
        mesh.AddScaleOp().Set(Gf.Vec3f(coordinate_unit_m))
        mesh.CreateFaceVertexCountsAttr([3]*len(piece['faces']))
        mesh.CreateFaceVertexIndicesAttr(piece['faces'].ravel().tolist())
        mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateVisibilityAttr('invisible')
        UsdPhysics.CollisionAPI.Apply(x).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(x).CreateApproximationAttr('convexHull')
        schemas=list(x.GetAppliedSchemas());schemas+=['PhysxCollisionAPI','PhysxConvexHullCollisionAPI']
        x.SetMetadata('apiSchemas',Sdf.TokenListOp.CreateExplicit(schemas))
        x.CreateAttribute('physxConvexHullCollision:hullVertexLimit',Sdf.ValueTypeNames.Int).Set(255)
        # The USD default is 1 mm, larger than the entire source pin diameter.
        # Preserve source dimensions rather than inflating a submillimetre hull.
        x.CreateAttribute('physxConvexHullCollision:minThickness',Sdf.ValueTypeNames.Float).Set(0.)
        for name in ('physxCollision:contactOffset','physxCollision:restOffset'):
            a=prim.GetAttribute(name)
            if a and a.HasAuthoredValueOpinion():x.CreateAttribute(name,a.GetTypeName()).Set(a.Get())
        if filters:UsdPhysics.FilteredPairsAPI.Apply(x).CreateFilteredPairsRel().SetTargets(filters)
        for name,targets in material_relations:x.CreateRelationship(name).SetTargets(targets)
        paths.append(path)
    core.GetFaceVertexCountsAttr().Set([3]*len(remaining));core.GetFaceVertexIndicesAttr().Set(remaining.ravel().tolist())
    return dict(scope='SOURCE_STEPPED_PIN_CONVEX_CONTACT_CANDIDATE; NOT YET PHYSICALLY_VALIDATED',
        core=core_path,pin_count=len(audits),piece_count=len(paths),paths=paths,
        original_core_face_count=len(f),remaining_core_face_count=len(remaining),
        pin_mesh_coordinate_unit_m=1e-6,
        source_pin_audits=audits,source_profile_vertices_used_without_scaling=True,
        preserved=['Body mass/COM/inertia','pin axial levels and circumferential source vertices','original materials and collision offsets','Socket geometry'],
        changed='Pin collision representation only: SDF pin surfaces replaced by three source convex pieces per stepped pin; the remaining SDF core is capped at the source root planes',
        native_cooked_geometry_verification_required=True)


def install_source_pin_local_sdfs(stage,*,body_path,core_path,source_pin_centres,group_at_same_spacing=False):
    """Same source pin solids; separate local SDFs instead of a shared core SDF."""
    core=UsdGeom.Mesh.Get(stage,core_path);prim=core.GetPrim()
    if len(UsdGeom.Xformable(core).GetOrderedXformOps()):raise ValueError('core-local transform is not identity')
    v=np.asarray(core.GetPointsAttr().Get(),float);f=np.asarray(core.GetFaceVertexIndicesAttr().Get(),int).reshape(-1,3)
    centres=np.asarray(json.load(Path(source_pin_centres).open())['source_plug_pin_centers_m'])
    remaining,_,audits,solids=source_pin_pieces(v,f,centres)
    groups=[[solid] for solid in solids]
    axial_extent=float(np.ptp(solids[0]['vertices'][:,2]))
    if group_at_same_spacing:
        # Pack neighbouring pins only while XY extent stays below the existing
        # pin axial extent. The longest SDF extent and resolution do not grow.
        remaining_ids=set(range(len(solids)));groups=[]
        bounds=np.array([[s['vertices'].min(0),s['vertices'].max(0)] for s in solids])
        while remaining_ids:
            ids=np.array(sorted(remaining_ids));best=[]
            for x in np.unique(bounds[ids,0,0]):
                possible=ids[(bounds[ids,0,0]>=x-1e-12)&(bounds[ids,1,0]<=x+axial_extent-1e-9)]
                for y in np.unique(bounds[possible,0,1]):
                    keep=possible[(bounds[possible,0,1]>=y-1e-12)&(bounds[possible,1,1]<=y+axial_extent-1e-9)]
                    if len(keep)>len(best):best=keep.tolist()
            if not best:raise ValueError('could not pack source pins without increasing SDF spacing')
            groups.append([solids[i] for i in best]);remaining_ids.difference_update(best)
    filters=UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets() if prim.HasAPI(UsdPhysics.FilteredPairsAPI) else []
    materials=[(rel.GetName(),rel.GetTargets()) for rel in prim.GetRelationships() if rel.GetName().startswith('material:binding')]
    paths=[]
    group_records=[]
    for group_index,group in enumerate(groups):
        combined=trimesh.util.concatenate([trimesh.Trimesh(s['vertices'],s['faces'],process=False) for s in group])
        solid={'vertices':np.asarray(combined.vertices),'faces':np.asarray(combined.faces)}
        extent=np.ptp(solid['vertices'],axis=0)
        if extent.max()>axial_extent+1e-9:raise ValueError('grouping increased longest SDF extent')
        path=f'{body_path}/SourcePinSdf_{group_index:03d}'
        if stage.GetPrimAtPath(path):raise ValueError('pin SDF already exists')
        mesh=UsdGeom.Mesh.Define(stage,path);x=mesh.GetPrim();centre=solid['vertices'].mean(axis=0)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy((solid['vertices']-centre).astype(np.float32)))
        mesh.AddTranslateOp().Set(Gf.Vec3d(*centre))
        mesh.CreateFaceVertexCountsAttr([3]*len(solid['faces']));mesh.CreateFaceVertexIndicesAttr(solid['faces'].ravel().tolist())
        mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateVisibilityAttr('invisible')
        UsdPhysics.CollisionAPI.Apply(x).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(x).CreateApproximationAttr('sdf')
        x.CreateAttribute('kcg:sourcePinIndices',Sdf.ValueTypeNames.IntArray).Set([s['pin'] for s in group])
        x.SetMetadata('apiSchemas',Sdf.TokenListOp.CreateExplicit(list(x.GetAppliedSchemas())+['PhysxCollisionAPI','PhysxSDFMeshCollisionAPI']))
        for name in ('physxCollision:contactOffset','physxCollision:restOffset','physxSDFMeshCollision:sdfResolution',
                     'physxSDFMeshCollision:sdfSubgridResolution','physxSDFMeshCollision:sdfNarrowBandThickness','physxSDFMeshCollision:sdfTriangleCountReductionFactor'):
            a=prim.GetAttribute(name)
            if a and a.HasAuthoredValueOpinion():x.CreateAttribute(name,a.GetTypeName()).Set(a.Get())
        if filters:UsdPhysics.FilteredPairsAPI.Apply(x).CreateFilteredPairsRel().SetTargets(filters)
        for name,targets in materials:x.CreateRelationship(name).SetTargets(targets)
        paths.append(path)
        group_records.append(dict(path=path,pin_indices=[s['pin'] for s in group],source_extent_m=extent.tolist()))
    core.GetFaceVertexCountsAttr().Set([3]*len(remaining));core.GetFaceVertexIndicesAttr().Set(remaining.ravel().tolist())
    return dict(scope='SOURCE_PIN_LOCAL_SDF_CANDIDATE_NOT_YET_VALIDATED',paths=paths,pin_count=len(solids),sdf_shape_count=len(paths),
        grouped_without_increasing_longest_sdf_extent=group_at_same_spacing,source_axial_extent_m=axial_extent,groups=group_records,
        preserved=['Original stepped pin mesh','Body mass/COM/inertia','materials','contact offsets','SDF resolution/subgrid/narrow band configuration','Socket mesh'],
        changed='The same source pin surfaces use their own local SDF bounds; root caps are internal to the unchanged source solid union',
        source_pin_audits=audits)


def install_source_pin_analytic_cylinders(stage,*,body_path,core_path,source_pin_centres,step_cylinder_audit):
    """Use original STEP pin cylinders, retaining the existing circular root stubs.

This deliberately targets the analytic STEP surface, not its inscribed
70/140-sided visualization. The bounded tessellation difference is reported.
Per-shape custom geometry avoids convex-hull cooking and global settings.
"""
    source=json.load(Path(source_pin_centres).open());analytic=json.load(Path(step_cylinder_audit).open())
    centres=np.asarray(source['source_plug_pin_centers_m']);step_rows=analytic['rows']
    step_centres=np.unique(np.round(np.array([r['axis_origin_m'][:2] for r in step_rows]),12),axis=0)
    if len(step_centres)!=128 or cKDTree(step_centres).query(centres)[0].max()>1e-9:
        raise ValueError('original STEP pin axes do not match all source model pins')
    radius=float(source['plug_pin_cylinder_radius_m']);zlo,zhi=source['plug_pin_cylinder_z_m']
    if any(abs(r['radius_m']-radius)>1e-12 or abs(r['bounds_m'][2]-zlo)>1e-9
           or abs(r['bounds_m'][5]-zhi)>1e-9 for r in step_rows):
        raise ValueError('STEP cylindrical dimensions disagree with the source model')
    core=UsdGeom.Mesh.Get(stage,core_path);prim=core.GetPrim()
    if len(UsdGeom.Xformable(core).GetOrderedXformOps()):raise ValueError('core-local transform is not identity')
    v=np.asarray(core.GetPointsAttr().Get(),float);f=np.asarray(core.GetFaceVertexIndicesAttr().Get(),int).reshape(-1,3)
    remaining,_,audits,solids=source_pin_pieces(v,f,centres)
    filters=UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets() if prim.HasAPI(UsdPhysics.FilteredPairsAPI) else []
    materials=[(rel.GetName(),rel.GetTargets()) for rel in prim.GetRelationships() if rel.GetName().startswith('material:binding')]
    paths=[];records=[]
    for solid in solids:
        i=solid['pin'];vv=solid['vertices'];low=float(vv[:,2].min())
        bottom=vv[np.abs(vv[:,2]-low)<1e-10];stub_radius=float(np.linalg.norm(bottom[:,:2]-centres[i],axis=1).mean())
        if np.max(abs(np.linalg.norm(bottom[:,:2]-centres[i],axis=1)-stub_radius))>2e-9:
            raise ValueError('retained source stub is not a circular profile')
        for part,(start,end,radius_here,sides) in enumerate([(low,zlo,stub_radius,140),(zlo,zhi,radius,70)]):
            path=f'{body_path}/SourcePinCylinder_{i:03d}_{part}'
            if stage.GetPrimAtPath(path):raise ValueError('pin cylinder already exists')
            cylinder=UsdGeom.Cylinder.Define(stage,path);x=cylinder.GetPrim()
            cylinder.CreateAxisAttr('Z');cylinder.CreateRadiusAttr(radius_here);cylinder.CreateHeightAttr(end-start)
            cylinder.AddTranslateOp().Set(Gf.Vec3d(float(centres[i,0]),float(centres[i,1]),(start+end)/2))
            cylinder.CreateVisibilityAttr('invisible');UsdPhysics.CollisionAPI.Apply(x).CreateCollisionEnabledAttr(True)
            x.SetMetadata('apiSchemas',Sdf.TokenListOp.CreateExplicit(list(x.GetAppliedSchemas())+['PhysxCollisionAPI']))
            x.CreateAttribute('physxCollision:customGeometry',Sdf.ValueTypeNames.Bool).Set(True)
            x.CreateAttribute('kcg:sourcePinIndex',Sdf.ValueTypeNames.Int).Set(i)
            for name in ('physxCollision:contactOffset','physxCollision:restOffset'):
                a=prim.GetAttribute(name)
                if a and a.HasAuthoredValueOpinion():x.CreateAttribute(name,a.GetTypeName()).Set(a.Get())
            if filters:UsdPhysics.FilteredPairsAPI.Apply(x).CreateFilteredPairsRel().SetTargets(filters)
            for name,targets in materials:x.CreateRelationship(name).SetTargets(targets)
            paths.append(path);records.append(dict(path=path,pin=i,part=part,radius_m=radius_here,z_range_body_m=[start,end],
                source='ORIGINAL_STEP_CYLINDRICAL_PIN' if part else 'EXISTING_CIRCULAR_RETAINED_STUB',
                maximum_inscribed_polygon_sagitta_m=float(radius_here*(1-np.cos(np.pi/sides)))))
    core.GetFaceVertexCountsAttr().Set([3]*len(remaining));core.GetFaceVertexIndicesAttr().Set(remaining.ravel().tolist())
    return dict(scope='ANALYTIC_SOURCE_CYLINDER_CANDIDATE_NOT_YET_VALIDATED',pin_count=128,paths=paths,pieces=records,
        source_step_audit=str(step_cylinder_audit),original_step_radius_m=radius,original_step_pin_z_m=[zlo,zhi],
        radius_or_length_fitted_to_success=False,source_mass_inertia_and_materials_changed=False,
        source_visual_mesh_changed=False,global_cylinder_approximation_setting_changed=False,
        native_custom_geometry_and_ray_verification_required=True,
        tessellation_difference='Analytic STEP circles replace inscribed source display polygons; same original nominal radii and axial dimensions. Existing root stubs retain their fitted circular source profile; submicrometre polygon sagitta is recorded.')
