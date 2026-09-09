"""Prepare independent rigid transform stacks without changing initial geometry.

This is an authoring-only diagnostic for the original robot. It does not move
objects after physics starts, change joint frames, or certify runtime behavior.
"""
import numpy as np


def author_independent_robot_rigid_frames(stage,root_path,*,physics_started):
    from pxr import Gf,Usd,UsdGeom,UsdPhysics
    if physics_started:raise RuntimeError("rigid transform stacks must be prepared before physics")
    root=stage.GetPrimAtPath(root_path)
    if not root:raise ValueError("robot root is missing")
    prims=list(Usd.PrimRange(root));cache=UsdGeom.XformCache(Usd.TimeCode.Default())
    world_before={str(p.GetPath()):cache.GetLocalToWorldTransform(p)
                  for p in prims if p.IsA(UsdGeom.Xformable)}
    def physics_and_bindings():
        return {str(p.GetPath()):{
            "attributes":{a.GetName():str(a.Get()) for a in p.GetAttributes()
                if a.HasAuthoredValueOpinion() and a.GetName().startswith(
                    ("physics:","physx","drive:","state:","newton"))},
            "relationships":{r.GetName():[str(v) for v in r.GetTargets()]
                for r in p.GetRelationships() if r.GetName().startswith(("physics:","physx","material:"))}}
            for p in prims}
    contract=physics_and_bindings();changed=[]
    for p in prims:
        if not p.HasAPI(UsdPhysics.RigidBodyAPI):continue
        parent=p.GetParent();rigid_parent=None
        while parent and parent!=stage.GetPseudoRoot():
            if parent.HasAPI(UsdPhysics.RigidBodyAPI):rigid_parent=parent;break
            parent=parent.GetParent()
        if rigid_parent is None:continue
        xform=UsdGeom.Xformable(p)
        if xform.GetResetXformStack():continue
        before=world_before[str(p.GetPath())]
        xform.MakeMatrixXform().Set(Gf.Matrix4d(before))
        xform.SetResetXformStack(True)
        changed.append({"prim":str(p.GetPath()),"rigid_parent":str(rigid_parent.GetPath())})
    cache.Clear();maximum=0.
    for path,prior in world_before.items():
        current=cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
        maximum=max(maximum,float(np.max(abs(np.asarray(current)-np.asarray(prior)))))
    if maximum>1e-10:raise RuntimeError(f"initial source world geometry changed: {maximum}")
    if physics_and_bindings()!=contract:raise RuntimeError("joint,material,mass or physics parameters changed")
    return {"scope":"PREPHYSICS_ROBOT_RIGID_TRANSFORM_REPRESENTATION_DIAGNOSTIC",
        "changed_rigid_frames":changed,"source_world_xform_count":len(world_before),
        "maximum_initial_world_matrix_change":maximum,
        "initial_world_geometry_preserved":True,"physics_and_material_bindings_preserved":True,
        "joint_paths_frames_drives_and_mass_inertia_preserved":True,
        "source_asset_modified":False,"post_start_pose_writes":False,
        "render_or_physics_improvement_validated":False,
        "rationale":"Test the documented independent transform-stack representation for nested rigid bodies; no solver/control/geometry replacement.",
        "primary_reference":"https://docs.omniverse.nvidia.com/kit/docs/asset-requirements/latest/capabilities/physics_bodies/physics_rigid_bodies/requirements/rigid-body-no-nesting.html"}
