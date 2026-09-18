"""Bounded speculative-contact margin for the source pins and rigid socket.

Only contactOffset changes. Rest offsets, vertices, materials, compliant sleeve
surfaces and control limits stay authored as before. This is a numerical runtime
candidate requiring the original physical acceptance, never an assembly claim.
"""
import re


def configure_pin_contact_margin(stage, contact_offset_m):
    from pxr import PhysxSchema, UsdPhysics

    value = float(contact_offset_m)
    if value != 1e-5:
        raise ValueError('This declared comparison uses a10micrometre speculative offset')
    pins = [p for p in stage.Traverse()
            if re.fullmatch(r'/World/TE_J35FreeSplitPlug/Body/OfficialPinWholeQuarter_\d{3}_[0-3]',str(p.GetPath()))]
    socket = stage.GetPrimAtPath('/World/TEVisualHandoff/FixedReceptaclePose/OfficialVisual/Geometry')
    if len(pins)!=512 or not socket:
        raise ValueError('Requires the original four convex quarters for each of128pins and source socket')
    rows=[]
    for prim in [*pins,socket]:
        if (not prim.HasAPI(UsdPhysics.CollisionAPI)
                or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not True):
            raise ValueError('Selected source collider is not enabled')
        api=PhysxSchema.PhysxCollisionAPI(prim)
        before=float(api.GetContactOffsetAttr().Get())
        rest=float(api.GetRestOffsetAttr().Get())
        if abs(before-5e-5)>1e-10 or rest!=0.:
            raise ValueError('The source contact margin or physical rest surface differs from the checked baseline')
        rows.append({'path':str(prim.GetPath()),'before_contact_offset_m':before,'rest_offset_m':rest})
    for prim,row in zip([*pins,socket],rows):
        api=PhysxSchema.PhysxCollisionAPI.Apply(prim)
        api.GetContactOffsetAttr().Set(value)
        row['after_contact_offset_m']=float(api.GetContactOffsetAttr().Get())
        if api.GetRestOffsetAttr().Get()!=0. or abs(row['after_contact_offset_m']-value)>1e-11:
            raise RuntimeError('The requested speculative margin was not authored without moving the rest surface')
    return {'scope':'SPECULATIVE_CONTACT_MARGIN_COMPARISON_NOT_ACCEPTANCE',
            'pin_source_socket_combined_margin_before_m':1e-4,
            'pin_source_socket_combined_margin_after_m':2e-5,
            'physical_rest_surface_changed':False,'mesh_vertices_or_materials_changed':False,
            'compliant_sleeve_margins_changed':False,'all_collision_pairs_remain_enabled_as_before':True,
            'online_contact_truth_used_for_robot_control':False,
            'solver_iterations_or_physics_dt_changed_by_this_helper':False,
            'original_key_and_force_and_complete_assembly_checks_required':True,'colliders':rows}
