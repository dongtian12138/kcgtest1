"""Use the unchanged fingertip CAD as SDF only for fingertip--Nut contact.

Other pairs retain the original convex decomposition. No rigid body, mass,
joint, material or visual mesh is replaced, and no collider is disabled.
"""
import numpy as np


def author_nut_only_fingertip_sdf(stage, robot_root, nut_root, *, links=("f1Link3", "f2Link2", "f3Link3")):
    import omni.timeline
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

    clock = omni.timeline.get_timeline_interface()
    if clock.is_playing() or clock.get_current_time() != 0.:
        raise RuntimeError("contact representation must be authored before physics")
    nut = stage.GetPrimAtPath(nut_root)
    if not nut.IsValid() or not nut.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError("the existing Nut rigid body is required")
    existing = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)
                and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
    non_nut = [str(p.GetPath()) for p in existing
               if not str(p.GetPath()).startswith(nut_root+"/") and str(p.GetPath()) != nut_root]
    if not links or any(link not in ("f1Link3","f2Link2","f3Link3") for link in links):
        raise ValueError('Only original terminal links may use the Nut-specific source SDF')
    records = []
    created = []
    for link in links:
        selected = [p for p in existing if str(p.GetPath()).startswith(robot_root+"/")
                    and "/"+link+"/" in str(p.GetPath())]
        if len(selected) != 1 or not selected[0].IsA(UsdGeom.Mesh):
            raise ValueError("expected one active source mesh per fingertip")
        original = selected[0]
        if UsdPhysics.MeshCollisionAPI(original).GetApproximationAttr().Get() != "convexDecomposition":
            raise ValueError("the comparison requires the original convex-decomposition collider")
        mesh = UsdGeom.Mesh(original)
        if UsdGeom.Xformable(original).GetOrderedXformOps():
            raise ValueError("source mesh local transform must remain identity under its existing parent")
        material, _ = UsdShade.MaterialBindingAPI(original).ComputeBoundMaterial("physics")
        if not material:
            raise ValueError("source fingertip physics material is missing")
        body = original
        while body.IsValid() and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        mass_names = ("physics:mass", "physics:centerOfMass", "physics:diagonalInertia", "physics:principalAxes")
        before = {name: str(body.GetAttribute(name).Get()) for name in mass_names}
        new_path = str(original.GetPath())+"_NutSourceSdf"
        if stage.GetPrimAtPath(new_path).IsValid():
            raise ValueError("Nut-specific source collider already exists")
        new = UsdGeom.Mesh.Define(stage, new_path)
        new.CreatePointsAttr(mesh.GetPointsAttr().Get())
        new.CreateFaceVertexCountsAttr(mesh.GetFaceVertexCountsAttr().Get())
        new.CreateFaceVertexIndicesAttr(mesh.GetFaceVertexIndicesAttr().Get())
        new.CreateSubdivisionSchemeAttr("none")
        new.CreateVisibilityAttr("invisible")
        prim = new.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("sdf")
        PhysxSchema.PhysxCollisionAPI.Apply(prim)
        PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
        for attr in original.GetAttributes():
            if attr.HasAuthoredValueOpinion() and attr.GetName().startswith(("physxCollision:", "physxSDFMeshCollision:")):
                prim.CreateAttribute(attr.GetName(),attr.GetTypeName()).Set(attr.Get())
        sdf=PhysxSchema.PhysxSDFMeshCollisionAPI(prim)
        sdf.CreateSdfResolutionAttr(1024)
        sdf.CreateSdfSubgridResolutionAttr(6)
        sdf.CreateSdfNarrowBandThicknessAttr(.002)
        PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(prim).CreateWeldToleranceAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,UsdShade.Tokens.strongerThanDescendants,"physics")
        original_pairs = UsdPhysics.FilteredPairsAPI.Apply(original).CreateFilteredPairsRel()
        prior = list(original_pairs.GetTargets())
        original_pairs.AddTarget(nut_root)
        if (mesh.GetPointsAttr().Get() != new.GetPointsAttr().Get()
                or mesh.GetFaceVertexIndicesAttr().Get() != new.GetFaceVertexIndicesAttr().Get()
                or before != {name: str(body.GetAttribute(name).Get()) for name in mass_names}):
            raise RuntimeError("source geometry or explicit mass properties changed")
        created.append(prim)
        records.append({"link":link,"original_collider":str(original.GetPath()),
            "nut_only_collider":new_path,"source_vertex_count":len(mesh.GetPointsAttr().Get()),
            "source_face_count":len(mesh.GetFaceVertexCountsAttr().Get()),
            "material":str(material.GetPath()),"prior_filtered_pairs":list(map(str,prior)),
            "sdf_resolution":prim.GetAttribute("physxSDFMeshCollision:sdfResolution").Get(),
            "source_geometry_and_explicit_mass_unchanged":True})
    new_paths = [str(p.GetPath()) for p in created]
    for prim in created:
        targets = list(dict.fromkeys(non_nut+[p for p in new_paths if p != str(prim.GetPath())]))
        UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().SetTargets(targets)
    return {"scope":"NUT_PAIR_ONLY_SOURCE_FINGERTIP_SDF_CONTACT_DIAGNOSTIC",
            "nut_rigid_body":nut_root,"pairs":records,
            "other_collision_pairs_use_original_decomposition":True,
            "all_original_colliders_remain_enabled":True,
            "mass_inertia_joints_contact_materials_and_visuals_changed":False,
            "physical_pair_filter_behavior_not_yet_validated":True,
            "online_truth_used":False,"post_start_authoring":False}
