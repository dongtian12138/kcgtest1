#!/usr/bin/env python3
"""Partition the source circumferential bulge for a future local contact model.

Offline geometry only. The exterior union, source keys and original assets are
preserved. This does not assign a spring law or install a new runtime collider.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import trimesh

from build_te_free_split_plug import _close_hidden_planar_boundaries


def cleaned(mesh):
    vertices, inverse = np.unique(mesh.vertices, axis=0, return_inverse=True)
    faces = inverse[mesh.faces]
    keep = ((faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2]))
    result = trimesh.Trimesh(vertices, faces[keep], process=False)
    if not result.is_volume or np.any(result.area_faces <= 0):
        raise ValueError("partition is not a closed positive nondegenerate volume")
    return result


def prepare(repository, output):
    output.mkdir(parents=True, exist_ok=False)
    source = repository / "artifacts/carts_grasp/CARTS_GRASP_V1/object_models/te_j35/plug_body_visual_mesh.npz"
    data = np.load(source)
    vertices, faces, caps = _close_hidden_planar_boundaries(data["vertices_m"], data["faces"])
    original = trimesh.Trimesh(vertices, faces, process=False)
    if not original.is_volume:
        raise ValueError("the existing source body closure is not a positive volume")
    # Place the internal split inside the 17.4117 mm CAD neck, avoiding
    # coincidence with its tessellated surface. The exterior union is unchanged;
    # this is not a supplier spring thickness or a change to the spring outline.
    base_radius = .01735
    # The preceding exact key-root plane created a collinear boolean face.
    # Put this interior partition 4.6 micrometres behind the source key roots;
    # all original key vertices must remain on the rigid partition below.
    z_min, z_max = -.0141, -.00765
    annulus = trimesh.creation.annulus(r_min=base_radius, r_max=.021,
                                      height=z_max-z_min, sections=140)
    annulus.apply_translation([0, 0, (z_min+z_max)/2])
    band = cleaned(trimesh.boolean.intersection([original, annulus], engine="manifold"))
    rigid = cleaned(trimesh.boolean.difference([original, annulus], engine="manifold"))
    union = cleaned(trimesh.boolean.union([rigid, band], engine="manifold"))
    volume_error = abs(rigid.volume + band.volume-original.volume)
    if volume_error > original.volume * 1e-6:
        raise ValueError("partition volume does not conserve the original body volume")
    # Deterministic original-surface samples, including every key vertex.
    raw, triangles = data["vertices_m"], data["faces"]
    slab = (raw[:, 2] >= -.007646) & (raw[:, 2] <= -.000761)
    outside = slab & (np.linalg.norm(raw[:, :2], axis=1) > .0183)
    key_faces = np.all(slab[triangles], axis=1) & np.any(outside[triangles], axis=1)
    keys = raw[np.unique(triangles[key_faces])]
    samples = np.vstack((raw[np.linspace(0, len(raw)-1, 2000).astype(int)], keys))
    _, union_distance, _ = trimesh.proximity.closest_point(union, samples)
    _, key_distance, _ = trimesh.proximity.closest_point(rigid, keys)
    if union_distance.max() > 5e-8 or key_distance.max() > 5e-8:
        raise ValueError("source exterior/key samples were displaced by the partition")
    meshes = {}
    for name, mesh in (("rigid_body", rigid), ("circumferential_band", band)):
        path = output / (name + ".npz")
        np.savez_compressed(path, vertices_m=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces, np.int32))
        meshes[name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            vertex_count=len(mesh.vertices), triangle_count=len(mesh.faces),
                            volume_m3=float(mesh.volume), bounds_m=mesh.bounds.tolist())
    result = dict(
        scope="OFFLINE_GEOMETRY_PARTITION_FOR_LOCAL_COMPLIANCE_DEVELOPMENT_ONLY",
        source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        original_asset_modified=False, installed_in_runtime=False,
        mass_inertia_joint_or_friction_changed=False, spring_law_assigned=False,
        component_identity="Circumferential bulge is consistent with the manufacturer's grounding-finger location; source STEP supplies a continuous simplified surface.",
        internal_split_surface="Representative cylindrical interior partition; not a supplier spring thickness or number of fingers.",
        internal_base_radius_m=base_radius, axial_partition_m=[z_min, z_max],
        source_rear_neck_radius_m=.0174117,
        original_hidden_caps=caps, original_volume_m3=float(original.volume),
        partition_volume_error_m3=float(volume_error),
        original_surface_sample_count=len(samples),
        maximum_sample_to_partition_union_distance_m=float(union_distance.max()),
        original_key_face_count=int(key_faces.sum()),
        maximum_key_vertex_to_rigid_partition_distance_m=float(key_distance.max()),
        meshes=meshes,
        unresolved=["effective stiffness and damping", "physical spring thickness/cuts", "contact-model resolution dependence", "other missing pin socket and seal details"],
        physical_assembly_or_compliance_validated=False)
    (output / "geometry_manifest.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


def prepare_sector_candidate(source_model, output, sector_count=8, component='grounding'):
    """Localize contact sampling; keep the complete existing band envelope.

    Equal angular sectors are a numerical partition, not a claim about the
    number or stiffness of the manufacturer's grounding fingers.
    """
    import manifold3d
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, Vt

    if sector_count != 8:
        raise ValueError('This bounded candidate uses eight sectors; no parameter sweep')
    source_model=source_model.resolve()
    output.mkdir(parents=True,exist_ok=False)
    source=Usd.Stage.Open(str(source_model))
    stage=Usd.Stage.Open(source.Flatten());layer=stage.GetRootLayer()
    if component not in ('grounding','socket_thread'):raise ValueError(component)
    band_path='/World/TE_J35FreeSplitPlug/Body/CompliantGroundingBandCollision'
    path=(band_path if component=='grounding' else
          '/World/TE_J35FreeSplitPlug/CouplingNut/SourceCadCollision')
    prim=stage.GetPrimAtPath(path);shape=UsdGeom.Mesh(prim)
    vertices=np.asarray(shape.GetPointsAttr().Get(),float)
    faces=np.asarray(shape.GetFaceVertexIndicesAttr().Get(),np.uint32).reshape(-1,3)
    source_extent=np.ptp(vertices,axis=0)
    source_resolution=int(prim.GetAttribute('physxSDFMeshCollision:sdfResolution').Get())
    source_spacing=float(source_extent.max()/source_resolution)
    source_band=prim.GetAttribute('physxSDFMeshCollision:sdfNarrowBandThickness').Get()
    source_band=.01 if source_band is None else float(source_band)
    source_margin=prim.GetAttribute('physxSDFMeshCollision:sdfMargin').Get()
    source_margin=.01 if source_margin is None else float(source_margin)
    solid=manifold3d.Manifold(manifold3d.Mesh(
        vert_properties=(vertices*1000).astype(np.float32),tri_verts=faces))
    if str(solid.status())!='Error.NoError':
        raise ValueError(f'Original band is not a valid volume: {solid.status()}')
    pieces=[];installed=[];mesh_audits=[]
    for index in range(sector_count):
        a,b=2*np.pi*np.asarray([index,index+1])/sector_count
        part=solid.split_by_plane([-np.sin(a),np.cos(a),0.],0.)[0]
        part=part.split_by_plane([np.sin(b),-np.cos(b),0.],0.)[0]
        if part.is_empty():raise ValueError('Empty band sector')
        pieces.append(part)
        mesh=part.to_mesh();v=np.asarray(mesh.vert_properties[:,:3],float)/1000
        f=np.asarray(mesh.tri_verts,np.int32)
        # USD/PhysX uses float32 points. Exact duplicate vertices introduced
        # by that conversion can otherwise leave zero-area cooking triangles.
        v,reverse=np.unique(v.astype(np.float32),axis=0,return_inverse=True)
        f=reverse[f];keep=(f[:,0]!=f[:,1])&(f[:,1]!=f[:,2])&(f[:,0]!=f[:,2])
        removed=int((~keep).sum());f=f[keep]
        check=trimesh.Trimesh(v.astype(float),f,process=False)
        if not check.is_volume or np.any(check.area_faces<=0):
            raise ValueError('Float32 sector must remain a closed positive nondegenerate volume')
        target=path+(f'_Sector_{index:02d}' if component=='grounding' else f'_SocketSector_{index:02d}')
        if not Sdf.CopySpec(layer,Sdf.Path(path),layer,Sdf.Path(target)):
            raise RuntimeError('Could not copy the original band material/filter settings')
        new=UsdGeom.Mesh.Get(stage,target)
        new.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
        new.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f),3,np.int32)))
        new.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.ravel()))
        extent=np.ptp(v.astype(float),axis=0)
        resolution=max(2,int(np.ceil(extent.max()/source_spacing)))
        new.GetPrim().GetAttribute('physxSDFMeshCollision:sdfResolution').Set(resolution)
        diagonal_ratio=float(np.linalg.norm(source_extent)/np.linalg.norm(extent))
        for name,value in [('sdfNarrowBandThickness',source_band*diagonal_ratio),
                           ('sdfMargin',source_margin*diagonal_ratio)]:
            new.GetPrim().CreateAttribute('physxSDFMeshCollision:'+name,Sdf.ValueTypeNames.Float).Set(value)
        mesh_audits.append({'path':target,'removed_exact_duplicate_vertex_faces':removed,
            'resolution':resolution,'actual_spacing_m':float(extent.max()/resolution),
            'source_spacing_m':source_spacing,'absolute_narrow_band_and_margin_preserved':True})
        if component=='socket_thread':
            # Retain the original full Nut contact surface for all hand/table
            # interactions. These sectors participate only against Socket.
            filters=UsdPhysics.FilteredPairsAPI(stage.GetPrimAtPath(band_path)).GetFilteredPairsRel().GetTargets()
            filters=[x for x in filters if str(x)!='/World/TE_J35FreeSplitPlug/CouplingNut']
            filters.append(Sdf.Path('/World/TE_J35FreeSplitPlug/Body'))
            UsdPhysics.FilteredPairsAPI.Apply(new.GetPrim()).CreateFilteredPairsRel().SetTargets(filters)
        installed.append(target)
    union=pieces[0]
    for part in pieces[1:]:union=union+part
    symmetric_difference=(union-solid).volume()+(solid-union).volume()
    if abs(symmetric_difference)>1e-5 or abs(sum(x.volume() for x in pieces)-solid.volume())>1e-5:
        raise RuntimeError('Band partition changed the source volume')
    um=union.to_mesh()
    union_mesh=trimesh.Trimesh(np.asarray(um.vert_properties[:,:3],float),um.tri_verts,process=False)
    _,dist,_=trimesh.proximity.closest_point(union_mesh,vertices*1000)
    if not np.isfinite(dist).all() or dist.max()>5e-5:
        raise RuntimeError('Source surface distance is invalid or moved more than 50 nm')
    if component=='grounding':
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    else:
        relation=UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel()
        relation.SetTargets(list(dict.fromkeys([*relation.GetTargets(),
            Sdf.Path('/World/TEVisualHandoff/FixedReceptaclePose/OfficialVisual/Geometry')])))
    candidate=output/'connector_model.usdc';stage.Flatten().Export(str(candidate))
    shutil.copy2(source_model.with_name('install_model.py'),output/'install_model.py')
    report=json.loads(source_model.with_name('connector_model_assembly_scene.json').read_text())
    audit={'scope':'UNVALIDATED_LOCAL_CONTACT_REPRESENTATION_CANDIDATE_NOT_ASSEMBLY_SUCCESS',
        'source_model':str(source_model),'component':component,'sector_count':sector_count,'installed_paths':installed,
        'original_collider_replaced_for_socket_contacts_only':path,
        'full_nut_hand_contact_preserved':component=='socket_thread',
        'source_volume_mm3':solid.volume(),'union_symmetric_difference_mm3':symmetric_difference,
        'maximum_original_vertex_to_union_distance_m':float(dist.max()/1000),
        'mass_inertia_joint_material_friction_and_per_contact_stiffness_changed':False,
        'purpose':'Distribute native contacts around the full original ring instead of a biased single manifold',
        'not_manufacturer_finger_count_or_force_calibration':True,
        'float32_mesh_and_grid_preparation':mesh_audits,
        'grid_rule_source':'https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/physxschema/class_physx_schema_physx_s_d_f_mesh_collision_a_p_i.html',
        'load_magnitude_and_partition_boundary_contacts_still_require_review':True}
    report['grounding_band_sector_candidate' if component=='grounding' else 'socket_thread_sector_candidate']=audit
    (output/'connector_model_assembly_scene.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'sector_geometry_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--source-model',type=Path,
                        help='Prepare the eight-sector repair candidate from a frozen model, offline only')
    parser.add_argument('--component',choices=('grounding','socket_thread'),default='grounding')
    arguments=parser.parse_args()
    if arguments.source_model:
        prepare_sector_candidate(arguments.source_model,arguments.output.resolve(),component=arguments.component)
    else:
        prepare(Path(__file__).resolve().parents[3],arguments.output.resolve())
