"""Declared storage-precision comparison for the Nut's external-contact SDF."""


def configure_nut_external_sdf_bits(stage, bits):
    from pxr import PhysxSchema, UsdPhysics

    if bits != 8:
        raise ValueError('This local comparison requests eight bits per sparse SDF value')
    path='/World/TE_J35FreeSplitPlug/CouplingNut/ExternalSurfaceContact'
    prim=stage.GetPrimAtPath(path)
    if (not prim or not prim.HasAPI(UsdPhysics.CollisionAPI)
            or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not True
            or prim.GetAttribute('physics:approximation').Get()!='sdf'):
        raise ValueError('Requires the original active external Nut SDF')
    api=PhysxSchema.PhysxSDFMeshCollisionAPI(prim)
    names=('physxSDFMeshCollision:sdfResolution','physxSDFMeshCollision:sdfSubgridResolution',
           'physxSDFMeshCollision:sdfNarrowBandThickness','physxCollision:contactOffset','physxCollision:restOffset')
    before={name:prim.GetAttribute(name).Get() for name in names}
    previous=str(api.GetSdfBitsPerSubgridPixelAttr().Get())
    if previous!='BitsPerPixel16' or before[names[0]]!=1024 or before[names[1]]!=6:
        raise ValueError('Unexpected source external Nut SDF layout')
    api.CreateSdfBitsPerSubgridPixelAttr('BitsPerPixel8')
    if before!={name:prim.GetAttribute(name).Get() for name in names}:
        raise RuntimeError('The SDF storage comparison changed another numerical parameter')
    return {'scope':'EXTERNAL_NUT_SPARSE_SDF_STORAGE_COMPARISON_REQUIRES_PHYSICAL_REVALIDATION',
            'path':path,'before':previous,'after':str(api.GetSdfBitsPerSubgridPixelAttr().Get()),
            'unchanged_parameters':before,'mesh_vertices_faces_material_mass_or_control_modified':False,
            'quantization_is_not_lossless':True,'same_accuracy_claimed':False,
            'original_key_force_mating_and_release_acceptance_required':True}
