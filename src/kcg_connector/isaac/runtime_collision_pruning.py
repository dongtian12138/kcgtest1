"""Omit disabled collision API instances from a runtime USD layer.

Meshes, transforms and all enabled colliders stay authored. Source files are
never saved. This only avoids initializing collision data that was explicitly
disabled before the optimization; it is not a collision simplification.
"""


def omit_disabled_collision_apis(stage, *, deactivate_invisible_leaves=False):
    from pxr import UsdGeom, UsdPhysics

    removed = []
    enabled_before = []
    for prim in list(stage.Traverse()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        if enabled is not False:
            enabled_before.append(str(prim.GetPath()))
            continue
        if not prim.GetAttribute('physics:collisionEnabled').HasAuthoredValueOpinion():
            raise RuntimeError('Only explicitly disabled collision instances may be omitted')
        row = {'path': str(prim.GetPath()),
               'approximation': prim.GetAttribute('physics:approximation').Get(),
               'collision_enabled_before': False}
        if not prim.RemoveAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError('The disabled collision API was not omitted')
        if prim.GetAttribute('physics:collisionEnabled').Get() is not False:
            raise RuntimeError('The original disabled attribute must stay authored')
        row['hidden_leaf_deactivated'] = False
        if (deactivate_invisible_leaves and not prim.GetChildren()
                and UsdGeom.Imageable(prim).ComputeVisibility() == 'invisible'
                and not prim.HasAPI(UsdPhysics.RigidBodyAPI)):
            prim.SetActive(False)
            row['hidden_leaf_deactivated'] = True
        removed.append(row)
    enabled_after = [str(p.GetPath()) for p in stage.Traverse()
                     if p.HasAPI(UsdPhysics.CollisionAPI)
                     and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False]
    if enabled_before != enabled_after:
        raise RuntimeError('Enabled collision roster changed')
    return {'scope': 'RUNTIME_OMISSION_OF_EXPLICITLY_DISABLED_COLLISION_APIS',
            'removed_disabled_apis': removed,
            'enabled_collider_count': len(enabled_before),
            'enabled_collider_roster_unchanged': True,
            'mesh_or_transform_or_mass_edits': False,
            'source_files_saved': False,
            'physical_trajectory_equivalence_requires_measurement': True}
